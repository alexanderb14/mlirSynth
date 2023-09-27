#include "analysis/PolyhedralAnalysis.h"
#include "execution/ArgUtils.h"
#include "execution/ArrayUtils.h"
#include "execution/Executor.h"
#include "execution/Lowering.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Tools/ParseUtilities.h"
#include "synthesis/Synthesizer.h"
#include "transforms/ChangeSizesPass.h"
#include "transforms/CleanupPass.h"
#include "transforms/CopyModifiedMemrefsPass.h"
#include "transforms/LoopDistributionPass.h"
#include "transforms/LoopOutlinePass.h"

#include "lhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/IR/register.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/dialect/Register.h"
#include "thlo/transforms/passes.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;
using namespace mlir;

std::vector<func::FuncOp> getFunctions(mlir::Operation *op,
                                       std::string attrName) {
  std::vector<func::FuncOp> functions;
  op->walk([&](func::FuncOp func) {
    if (attrName.empty() || func->getAttr(attrName))
      functions.push_back(func);
  });
  return functions;
}

func::FuncOp lowerHLO(func::FuncOp &func) {
  auto *ctx = func->getContext();

  auto pm = std::make_shared<mlir::PassManager>(ctx);
  HLO::addCHLOToAffinePasses(pm);

  auto hloModule = createModule(*ctx, &func).release();

  if (failed(pm->run(hloModule))) {
    assert(false && "Couldn't lower to HLO to affine dialect");
  }

  return unwrapModule(hloModule).release();
}

bool testValidate(func::FuncOp lhsFunction, func::FuncOp rhsFunction,
                  bool printArgsAndResults = false, bool printResults = false) {
  auto *ctx = lhsFunction->getContext();

  lhsFunction->setAttr("llvm.emit_c_interface", UnitAttr::get(ctx));
  lhsFunction.setSymName("foo");

  auto lhsModuleRef = createModule(*ctx, &lhsFunction);
  auto lhsModule = lhsModuleRef.release();

  // Create inputs.
  auto args = createArgs(lhsFunction);
  randomlyInitializeArgs(lhsFunction, args);
  auto targetShape = getReturnShape(lhsFunction);

  // Lower and run the lhs function on the inputs.
  auto pm = std::make_shared<mlir::PassManager>(ctx);
  Polygeist::addAffineToLLVMPasses(pm);
  assert(succeeded(pm->run(lhsModule)) &&
         "Failed to lower affine to LLVM dialect");

  auto refRet = getOwningMemRefForShape(targetShape);
  assert(succeeded(jitAndInvoke(lhsModule, args, refRet)));
  double *refOut = getReturnDataPtr(refRet);

  if (printArgsAndResults)
    printArgsAndResultsInPython(args, refOut, targetShape);

  // RHS
  rhsFunction->setAttr("llvm.emit_c_interface", UnitAttr::get(ctx));
  rhsFunction.setSymName("foo");
  auto rhsModuleRef = createModule(*ctx, &rhsFunction);
  auto rhsModule = rhsModuleRef.release();

  // Lower and run the rhs function on the inputs.
  auto pmRHS = std::make_shared<mlir::PassManager>(ctx);
  HLO::addAffineToLLVMPasses(pmRHS);
  assert(succeeded(pm->run(rhsModule)) &&
         "Failed to lower chlo to LLVM dialect");

  auto rhsRet = getOwningMemRefForShape(targetShape);
  convertScalarToMemrefArgs(args);
  assert(succeeded(jitAndInvoke(rhsModule, args, rhsRet)));
  double *rhsOut = getReturnDataPtr(rhsRet);

  if (printResults) {
    printArray(refOut, targetShape, llvm::outs());
    llvm::outs() << "\n";
    printArray(rhsOut, targetShape, llvm::outs());
    llvm::outs() << "\n";
  }

  // Test for equivalence.
  return areArraysEqual(refOut, rhsOut, targetShape);
}

int main(int argc, char **argv) {
  // Parse command line arguments.
  cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input file>"),
                                     cl::init("-"));
  cl::opt<bool> printArgsAndResults("print-args-and-results",
                                    cl::desc("Print args and results"),
                                    cl::init(false));
  cl::opt<bool> printResults("print-results", cl::desc("Print results"),
                             cl::init(false));
  cl::ParseCommandLineOptions(argc, argv, "Test Executor\n");

  // Initialize LLVM.
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  // Register dialects and passes.
  DialectRegistry registry;
  registerAllDialects(registry);
  registerLLVMDialectTranslation(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);

  registerAllPasses();
  mlir::hlo::registerLMHLOTransformsPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();
  mlir::thlo::registerAllThloPasses();

  // Create a context.
  MLIRContext ctx;
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();

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
  ParserConfig config(&ctx, /*verifyAfterParse=*/true, &fallbackResourceMap);
  OwningOpRef<Operation *> inputOp =
      parseSourceFileForTool(sourceMgr, config, /*insertImplicitModule*/ false);
  assert(inputOp && "Failed to parse input file");

  // Load original function module (in affine).
  auto originalFunctions = getFunctions(inputOp.get(), "irsynth.original");
  assert(originalFunctions.size() == 1 &&
         "Expected one function with the irsynth.original attribute");
  auto originalFunction = originalFunctions[0];

  // Load HLO function module(s).
  auto hloFunctions = getFunctions(inputOp.get(), "irsynth.raised");
  for (auto hloFunction : hloFunctions) {
    auto lowered = lowerHLO(hloFunction);

    originalFunction.dump();
    lowered->dump();

    bool equiv = testValidate(originalFunction, lowered,
                              printArgsAndResults, printResults);

    // Print the results.
    llvm::outs() << hloFunction.getName().str() << ": ";
    if (equiv) {
      llvm::outs() << "\033[1;42mResults are equal.\033[0m";
    } else {
      llvm::outs() << "\033[1;41mResults are different.\033[0m";
    }
    llvm::outs() << "\n";
  }
}
