#include "analysis/PolyhedralAnalysis.h"
#include "execution/ArgUtils.h"
#include "execution/ArrayUtils.h"
#include "execution/Executor.h"
#include "execution/Lowering.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
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

bool checkValidate(func::FuncOp lhsFunction, func::FuncOp rhsFunction,
                  bool printArgsAndResults = false, bool printResults = false) {
  auto *ctx = lhsFunction->getContext();
  auto pm = std::make_shared<mlir::PassManager>(ctx);

  auto builder = OpBuilder(ctx);

  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(ctx)));
  auto &moduleBlock = module->getRegion().getBlocks().front();
  builder.setInsertionPoint(&moduleBlock, moduleBlock.begin());

  // Rename and copy over functions.
  auto lhsFunctionClone = lhsFunction.clone();
  lhsFunctionClone.setSymName("lhs");
  module->push_back(lhsFunctionClone);

  auto rhsFunctionClone = rhsFunction.clone();
  rhsFunctionClone.setSymName("rhs");
  module->push_back(rhsFunctionClone);

  // Create a main function.
  auto mainFunc = builder.create<func::FuncOp>(UnknownLoc::get(ctx), "main",
                                           builder.getFunctionType({}, {}));

  auto &bodyBlock = *mainFunc.addEntryBlock();
  builder.setInsertionPoint(&bodyBlock, bodyBlock.begin());

  // Create operands.
  SmallVector<mlir::Value> operands;
  for (auto arg : lhsFunction.getArguments()) {
    if (arg.getType().isa<ShapedType>()) {
      auto memreftype = arg.getType().cast<MemRefType>();
      auto memref = builder.create<memref::AllocOp>(UnknownLoc::get(ctx), memreftype);
      operands.push_back(memref);

      auto afterLastOperand = builder.saveInsertionPoint();

      SmallVector<mlir::Value> indices;
      auto argShape = arg.getType().cast<ShapedType>().getShape();
      for (auto dimSize : argShape) {
        // Create for op.
        auto forOp = builder.create<scf::ForOp>(
            UnknownLoc::get(ctx),
            builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), 0),
            builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), dimSize),
            builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), 1));

        indices.push_back(forOp.getInductionVar());

        // Set insertion point inside body for next loop to be inserted.
        builder.setInsertionPointToStart(forOp.getBody());

        // If we are in the innermost loop, create the declaration.
        if (indices.size() == argShape.size()) {
          // Create decl.
          SmallVector<mlir::Value> operands = {};
          auto declOp = builder.create<func::CallOp>(UnknownLoc::get(ctx), 
                                       builder.getF64Type(),
                                       "cbmc_declare", operands);

          // Create store.
          builder.create<memref::StoreOp>(UnknownLoc::get(ctx),
                                          declOp->getResult(0),
                                          memref.getMemref(), indices);
        }
      }

      builder.restoreInsertionPoint(afterLastOperand);

    } else if (arg.getType().isa<FloatType>()) {
      operands.push_back(builder.create<arith::ConstantOp>(
          UnknownLoc::get(ctx), builder.getF64Type(),
          builder.getF64FloatAttr(0.0f)));
    } else {
      llvm::outs() << "Type: " << arg.getType() << "\n";
      assert(false && "Unsupported type");
    }
  }

  // Call functions on the created operands.
  auto lhsCallOp = builder.create<func::CallOp>(
      UnknownLoc::get(ctx), lhsFunctionClone.getResultTypes(),
      lhsFunctionClone.getSymName(), operands);
  auto rhsCallOp = builder.create<func::CallOp>(
      UnknownLoc::get(ctx), rhsFunctionClone.getResultTypes(),
      rhsFunctionClone.getSymName(), operands);

  auto afterLastCallOp = builder.saveInsertionPoint();

  // Assert element-wise equality.
  SmallVector<mlir::Value> indices;
  auto shape = lhsFunction.getResultTypes()[0].cast<ShapedType>().getShape();
  for (auto dimSize : shape) {
    // Create for op.
    auto forOp = builder.create<scf::ForOp>(
        UnknownLoc::get(ctx),
        builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), 0),
        builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), dimSize),
        builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), 1));

    indices.push_back(forOp.getInductionVar());

    // Set insertion point inside body for next loop to be inserted.
    builder.setInsertionPointToStart(forOp.getBody());

    // If we are in the innermost loop, create the equality check.
    if (indices.size() == shape.size()) {
      // Load from memrefs.
      auto lhsMemref = lhsCallOp.getResult(0);
      mlir::Type lhsType =
          lhsMemref.getType().cast<MemRefType>().getElementType();
      auto lhsLoad = builder.create<memref::LoadOp>(
          UnknownLoc::get(ctx), lhsType, lhsMemref, indices);

      auto rhsMemref = rhsCallOp.getResult(0);
      mlir::Type rhsType =
          rhsMemref.getType().cast<MemRefType>().getElementType();
      auto rhsLoad = builder.create<memref::LoadOp>(
          UnknownLoc::get(ctx), rhsType, rhsMemref, indices);

      // Create check.
      SmallVector<mlir::Value> checkOperands = {lhsLoad, rhsLoad};
      builder.create<func::CallOp>(UnknownLoc::get(ctx), 
                                   builder.getF64Type(),
                                   "cbmc_assert", checkOperands);
    }
  }

  // Must have the check function declaration.
  builder.setInsertionPoint(&moduleBlock, moduleBlock.begin());
  func::FuncOp cbmcAssertFwdDecl = builder.create<func::FuncOp>(
      UnknownLoc::get(ctx), "cbmc_assert",
      mlir::FunctionType::get(ctx, {builder.getF64Type(), builder.getF64Type()},
                              {builder.getF64Type()}));
  cbmcAssertFwdDecl.setPrivate();

  func::FuncOp cbmcDeclareFwdDecl = builder.create<func::FuncOp>(
      UnknownLoc::get(ctx), "cbmc_declare",
      mlir::FunctionType::get(ctx, {}, {builder.getF64Type()}));
  cbmcDeclareFwdDecl.setPrivate();

  // Must have a return op.
  builder.restoreInsertionPoint(afterLastCallOp);
  builder.create<func::ReturnOp>(UnknownLoc::get(ctx));
  module->dump();

  return true;
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
    checkValidate(originalFunction, lowered,
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
