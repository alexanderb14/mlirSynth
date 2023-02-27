#include "analysis/PolyhedralAnalysis.h"
#include "enumeration/Enumerator.h"
#include "execution/ArgUtils.h"
#include "execution/ArrayUtils.h"
#include "execution/Executor.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Tools/ParseUtilities.h"
#include "transforms/CleanupPass.h"
#include "transforms/CopyModifiedMemrefsPass.h"
#include "transforms/LoopDistributionPass.h"
#include "transforms/LoopOutlinePass.h"
#include "transforms/MemrefMinifyPass.h"

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

int main(int argc, char **argv) {
  // Parse command line arguments.
  cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input file>"),
                                     cl::init("-"));
  cl::opt<bool> printArgsAndResults("print-args-and-results",
                                    cl::desc("Print args and results"),
                                    cl::init(false));
  cl::opt<bool> printResults("print-results", cl::desc("Print results"),
                             cl::init(false));
  cl::ParseCommandLineOptions(argc, argv, "MLIR enumerator\n");

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

  originalFunction->setAttr("llvm.emit_c_interface", UnitAttr::get(&ctx));
  originalFunction.setSymName("foo");

  auto originalModuleRef = createModule(ctx, &originalFunction);
  auto originalModule = originalModuleRef.release();

  // Create inputs.
  auto args = createArgs(originalFunction);
  randomlyInitializeArgs(originalFunction, args);
  auto targetShape = getReturnShape(originalFunction);

  // Lower and run the original function on the inputs.
  auto executor = std::make_shared<Executor>(&ctx);

  assert(succeeded(executor->lowerAffineToLLVMDialect(originalModule)) &&
         "Failed to lower affine to LLVM dialect");
  auto refRet = getOwningMemRefForShape(targetShape);
  assert(succeeded(jitAndInvoke(originalModule, args, refRet, false)));
  double *refOut = getReturnDataPtr(refRet);

  if (printArgsAndResults)
    printArgsAndResultsInPython(args, refOut, targetShape);


  // Load HLO function module(s).
  auto hloFunctions = getFunctions(inputOp.get(), "irsynth.raised");
  for (auto hloFunction : hloFunctions) {
    auto functionName = hloFunction.getName().str();

    hloFunction->setAttr("llvm.emit_c_interface", UnitAttr::get(&ctx));
    hloFunction.setSymName("foo");
    auto hloModuleRef = createModule(ctx, &hloFunction);
    auto hloModule = hloModuleRef.release();

    // Lower and run the hlo function on the inputs.
    assert(succeeded(executor->lowerCHLOToLLVMDialect(hloModule)) &&
           "Failed to lower chlo to LLVM dialect");
    auto hloRet = getOwningMemRefForShape(targetShape);
    convertScalarToMemrefArgs(args);
    assert(succeeded(jitAndInvoke(hloModule, args, hloRet, false)));
    double *hloOut = getReturnDataPtr(hloRet);

    // Print the results.
    llvm::outs() << functionName << ": ";
    if (areArraysEqual(refOut, hloOut, targetShape)) {
      llvm::outs() << "\033[1;42mResults are equal.\033[0m";
    } else {
      llvm::outs() << "\033[1;41mResults are different.\033[0m";
    }
    llvm::outs() << "\n";

    if (printResults) {
      printArray(refOut, targetShape, llvm::outs());
      llvm::outs() << "\n";
      printArray(hloOut, targetShape, llvm::outs());
      llvm::outs() << "\n";
    }
  }
}
