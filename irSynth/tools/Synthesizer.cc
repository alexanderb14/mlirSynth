#include "ContextManager.h"

#include "Common.h"
#include "enumeration/ArgTuples.h"
#include "enumeration/Candidate.h"
#include "enumeration/Enumerator.h"
#include "enumeration/Guide.h"
#include "execution/Executor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "transforms/CopyModifiedMemrefsPass.h"
#include "transforms/LoopDistributionPass.h"
#include "transforms/LoopOutlinePass.h"
#include "transforms/MemrefMinifyPass.h"

#include "lhlo/IR/lhlo_ops.h"
#include "lhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/IR/register.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/ParseUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Register.h"
#include "thlo/transforms/passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"

#include <iterator>

using namespace llvm;
using namespace mlir;

std::vector<std::string> splitString(std::string &str) {
  std::vector<std::string> vect;
  std::string delimiter = ",";

  std::stringstream ss(str);
  while (ss.good()) {
    std::string substr;
    getline(ss, substr, ',');
    vect.push_back(substr);
  }

  return vect;
}

std::vector<func::FuncOp> getFunctions(mlir::Operation *op,
                                       std::string attrName) {
  std::vector<func::FuncOp> functions;
  op->walk([&](func::FuncOp func) {
    if (attrName.empty() || func->getAttr(attrName))
      functions.push_back(func);
  });
  return functions;
}

std::vector<RegisteredOperationName>
getDialectOps(MLIRContext *ctx, std::vector<Dialect *> &dialects,
              const std::vector<std::string> &ops = {}, bool printOps = false) {
  std::unordered_map<std::string, bool> opsMap;
  for (auto &op : ops) {
    opsMap[op] = true;
  }

  std::vector<RegisteredOperationName> opNames;
  for (auto *dialect : dialects) {
    for (auto op : ctx->getRegisteredOperations()) {
      if (&op.getDialect() == dialect) {
        if (opsMap.empty() ||
            opsMap.find(op.getIdentifier().str()) != opsMap.end()) {
          opNames.push_back(op);
        }
      }
    }
  }

  if (printOps) {
    llvm::outs() << "Registered ops:"
                 << "\n--------\n";
    for (auto opName : opNames) {
      opName.dump();
      llvm::outs() << "\n";
    }
  }

  return opNames;
}

int main(int argc, char **argv) {
  // Parse command line arguments.
  cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input file>"),
                                     cl::init("-"));

  cl::opt<bool> printStatusNames(
      "print-status-names", cl::desc("Print status names"), cl::init(false));
  cl::opt<bool> printStatusTiles(
      "print-status-tiles", cl::desc("Print status tiles"), cl::init(false));

  cl::opt<bool> printValidCandidates("print-valid-candidates",
                                     cl::desc("Print valid candidates"),
                                     cl::init(false));
  cl::opt<bool> printInvalidCandidates("print-invalid-candidates",
                                       cl::desc("Print invalid candidates"),
                                       cl::init(false));

  cl::opt<bool> printErrors("print-errors", cl::desc("Print errors"),
                            cl::init(false));
  cl::opt<bool> printStats("print-stats", cl::desc("Print stats"),
                           cl::init(false));

  cl::opt<std::string> ops(
      "ops", cl::desc("Comma separated list of allowed ops"), cl::init(""));
  cl::opt<int> maxNumOps("max-num-ops", cl::desc("Max number of operations"),
                         cl::init(3));

  cl::opt<int> numThreads("num-threads", cl::desc("Number of threads"),
                          cl::init(1));

  cl::opt<bool> ignoreEquivalentCandidates(
      "ignore-equivalent-candidates",
      cl::desc("Ignore computationally equivalent candidates"),
      cl::init(false));
  cl::opt<bool> guide("guide", cl::desc("Use guide to select allowed ops"),
                      cl::init(false));
  cl::opt<bool> distribute(
      "distribute",
      cl::desc("Distribute loops to split synthesis into smaller subproblems"),
      cl::init(false));

  cl::ParseCommandLineOptions(argc, argv, "MLIR enumerator\n");

  // Initialize LLVM.
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  // Initialize MLIR.
  ContextManagerPtr contextManager =
      std::make_shared<ContextManager>(printErrors);
  auto *ctx = contextManager->createContext();

  Dialect *hloDialect = ctx->getOrLoadDialect<mhlo::MhloDialect>();
  Dialect *chloDialect = ctx->getOrLoadDialect<chlo::ChloDialect>();
  std::vector<Dialect *> dialects = {hloDialect, chloDialect};

  // Parse the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());

  FallbackAsmResourceMap fallbackResourceMap;
  ParserConfig config(ctx, /*verifyAfterParse=*/true, &fallbackResourceMap);
  OwningOpRef<Operation *> inputOp =
      parseSourceFileForTool(sourceMgr, config, /*insertImplicitModule*/ false);
  assert(inputOp && "Failed to parse input file");

  // Transform the input op to prepare for synthesis.
  mlir::PassManager pm(ctx);
  if (distribute)
    pm.addPass(createLoopDistributionPass());
  pm.addPass(createMemrefMinifyPass());
  pm.addPass(createLoopOutlinePass());
  pm.addPass(createCopyModifiedMemrefsPass());
  if (failed(pm.run(inputOp.get()))) {
    llvm::errs() << "Failed to run passes on input file\n";
    return 1;
  }
  LLVM_DEBUG(llvm::dbgs() << "After preparation transformations:\n");
  LLVM_DEBUG(inputOp.get()->print(llvm::dbgs()));

  // Parse the funcion ops.
  std::vector<func::FuncOp> functions =
      getFunctions(inputOp.get(), "irsynth.original");
  LLVM_DEBUG(llvm::dbgs() << "Found " << functions.size()
                          << " functions to synthesize\n");

  IExecutorPtr executor;
  if (numThreads == 1) {
    ctx->disableMultithreading();
    executor = std::make_shared<Executor>(ctx);
  } else {
    executor = std::make_shared<ThreadedExecutor>(contextManager, numThreads);
  }

  // Synthesize functions.
  std::vector<std::string> supportedOps = {"chlo.broadcast_divide",
                                           "chlo.broadcast_add",
                                           "chlo.broadcast_subtract",
                                           "chlo.broadcast_multiply",
                                           "mhlo.dot",
                                           "mhlo.reduce",
                                           "mhlo.dynamic_reshape",
                                           "mhlo.dot_general"};
  llvm::DenseMap<func::FuncOp, OwningOpRef<ModuleOp>> originalToSynthesizedFns;
  llvm::DenseMap<func::FuncOp, std::vector<unsigned>>
      originalToSynthesizedArgIds;
  for (auto inputFuncOrig : functions) {
    auto inputFunc = inputFuncOrig.clone();
    // Get ops.
    std::vector<std::string> opsVec;
    if (guide) {
      opsVec = predictOps(supportedOps, inputFunc);
    } else if (!ops.empty()) {
      opsVec = splitString(ops);
    } else {
      opsVec = supportedOps;
    }
    auto availableOps = getDialectOps(ctx, dialects, opsVec, false);

    // Synthesize.
    LLVM_DEBUG(llvm::dbgs()
               << "Synthesizing function " << inputFunc.getName() << "\n");
    LLVM_DEBUG(inputFunc.print(llvm::dbgs()));

    CandidateStorePtr candidateStore = std::make_shared<CandidateStore>();

    EnumerationOptions options;
    options.printStatusNames = printStatusNames;
    options.printStatusTiles = printStatusTiles;
    options.printValidCandidates = printValidCandidates;
    options.printInvalidCandidates = printInvalidCandidates;
    options.printStats = printStats;
    options.maxNumOps = maxNumOps;
    options.ignoreEquivalentCandidates = ignoreEquivalentCandidates;

    ModuleAndArgIds enumerated = enumerateCandidates(
        *ctx, executor, inputFunc, candidateStore, availableOps, options);
    auto module = std::move(std::get<0>(enumerated));
    auto argIds = std::get<1>(enumerated);

    if (!module) {
      llvm::errs() << "Failed to synthesize function " << inputFunc.getName()
                   << "\n";
      return 1;
    }
    originalToSynthesizedFns[inputFuncOrig] = std::move(module);
    originalToSynthesizedArgIds[inputFuncOrig] = argIds;

    if (options.printStats) {
      candidateStore->dumpSizes();
    }
  }

  OpBuilder builder(ctx);

  for (auto &kv : originalToSynthesizedFns) {
    auto inputFunc = kv.first;
    auto synthesizedFunc = getFunctions(kv.second.get(), "irsynth.raised")[0];
    auto argIds = originalToSynthesizedArgIds[inputFunc];

    synthesizedFunc.setName(
        StringAttr::get(ctx, inputFunc.getName() + "_raised"));

    // Insert the synthesized functions into the original module.
    synthesizedFunc->moveAfter(inputFunc);

    // Get the call sites of the original function.
    std::vector<func::CallOp> callSites;
    inputOp.get()->walk([&](func::CallOp callOp) {
      if (callOp.getCallee() == inputFunc.getName())
        callSites.push_back(callOp);
    });

    // Change the called function to the synthesized function and
    // insert conversion ops for their arguments and results.
    for (auto callSite : callSites) {
      callSite.setCallee(synthesizedFunc.getName());

      // Arguments: Memref to tensor.
      std::vector<mlir::Value> callSiteOperands;
      for (auto operand : callSite.getOperands())
        callSiteOperands.push_back(operand);

      unsigned operandIdx = 0;
      for (operandIdx = 0; operandIdx < argIds.size(); ++operandIdx) {
        unsigned argId = argIds[operandIdx];

        auto callArg = callSiteOperands[argId];
        auto synthesizedArg = synthesizedFunc.getArgument(operandIdx);

        // If types are the same, no need to insert conversion.
        if (callArg.getType() == synthesizedArg.getType())
          continue;

        // If call site type is not a memref, no need to insert conversion.
        if (!callArg.getType().isa<MemRefType>())
          continue;

        // Insert bufferization.to_tensor ops for the call arguments.
        builder.setInsertionPoint(callSite);
        auto toTensorOp = builder.create<bufferization::ToTensorOp>(
            callSite.getLoc(), callArg);

        // Replace the call site argument with the result of the to_tensor op.
        callSite.setOperand(operandIdx, toTensorOp.getResult());
      }
      // Remove the remaining call site arguments.
      for (; operandIdx < callSite.getNumOperands(); ++operandIdx) {
        callSite->eraseOperand(operandIdx);
      }


      // Results: Tensor to memref.
      assert(callSite.getNumResults() == synthesizedFunc.getNumResults() &&
             "Number of call site results and function results must match");
      for (unsigned resultIdx = 0; resultIdx < callSite.getNumResults();
           resultIdx++) {
        auto callResType = callSite.getResultTypes()[resultIdx];
        auto synthesizedResType = synthesizedFunc.getResultTypes()[resultIdx];

        // If types are the same, no need to insert conversion.
        if (callResType == synthesizedResType)
          continue;

        // If call site type is not a memref, no need to insert conversion.
        if (!callResType.isa<MemRefType>())
          continue;

        // Get call site result.
        auto callRes = callSite.getResult(resultIdx);

        // Insert bufferization.to_memref ops for the call results.
        builder.setInsertionPointAfter(callSite);
        auto toMemrefOp = builder.create<bufferization::ToMemrefOp>(
            UnknownLoc::get(ctx), callResType, callRes);

        // Set the result type of the call site to the type of the synthesized
        // function result.
        callSite.getResult(resultIdx).setType(synthesizedResType);

        // Replace the uses of the call site result with the result of the
        // to_memref op.
        callRes.replaceAllUsesExcept(toMemrefOp.getResult(), toMemrefOp);
      }
    }
  }

  inputOp.get()->print(llvm::outs());

  return 0;
}
