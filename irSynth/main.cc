#include "ContextManager.h"
#include "Utils.h"
#include "transforms/LoopDistributionPass.h"
#include "enumeration/ArgTuples.h"
#include "enumeration/Candidate.h"
#include "enumeration/Enumerator.h"
#include "execution/Executor.h"

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

  // Get ops.
  std::vector<std::string> opsVec;
  if (ops.empty()) {
    opsVec = {"chlo.broadcast_divide",
              "chlo.broadcast_add",
              "chlo.broadcast_subtract",
              "chlo.broadcast_multiply",
              "mhlo.dot",
              "mhlo.reduce",
              "mhlo.dynamic_reshape",
              "mhlo.dot_general"};
  } else {
    opsVec = splitString(ops);
  }

  Dialect *hloDialect = ctx->getOrLoadDialect<mhlo::MhloDialect>();
  Dialect *chloDialect = ctx->getOrLoadDialect<chlo::ChloDialect>();
  std::vector<Dialect *> dialects = {hloDialect, chloDialect};
  auto availableOps = getDialectOps(ctx, dialects, opsVec, true);

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

  // Run passes.
  mlir::PassManager pm(ctx);
  //pm.addNestedPass<mlir::func::FuncOp>(createLoopDistributionPass());
  if (failed(pm.run(inputOp.get()))) {
    llvm::errs() << "Failed to run passes on input file\n";
    return 1;
  }

  // Parse the funcion ops.
  std::vector<func::FuncOp> functions = getFunctions(inputOp.get());
  func::FuncOp inputFunction = functions[0];

  // Synthesis.
  IExecutorPtr executor;
  if (numThreads == 1) {
    ctx->disableMultithreading();
    executor = std::make_shared<Executor>(ctx);
  } else {
    executor = std::make_shared<ThreadedExecutor>(contextManager, numThreads);
  }

  CandidateStorePtr candidateStore = std::make_shared<CandidateStore>();

  EnumerationOptions options;
  options.printStatusNames = printStatusNames;
  options.printStatusTiles = printStatusTiles;
  options.printValidCandidates = printValidCandidates;
  options.printInvalidCandidates = printInvalidCandidates;
  options.printStats = printStats;
  options.maxNumOps = maxNumOps;
  options.ignoreEquivalentCandidates = ignoreEquivalentCandidates;

  bool status = enumerateCandidates(*ctx, executor, inputFunction,
                                    candidateStore, availableOps, options);

  candidateStore->dumpSizes();

  if (status)
    return 0;
  return 1;
}
