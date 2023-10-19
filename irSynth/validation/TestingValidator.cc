#include "TestingValidator.h"

#include "execution/ArrayUtils.h"
#include "execution/Lowering.h"
#include "synthesis/Synthesizer.h"

using namespace mlir;

bool testValidate(func::FuncOp lhsFunction, func::FuncOp rhsFunction,
                  bool printArgsAndResults, bool printResults) {
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
