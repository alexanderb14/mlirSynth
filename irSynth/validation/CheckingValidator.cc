#include "CheckingValidator.h"

#include "mlir/IR/Location.h"
#include "transforms/MemrefCopyToLoopsPass.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

OwningOpRef<ModuleOp> buildModule(func::FuncOp lhsFunction,
                                  func::FuncOp rhsFunction) {
  auto *ctx = lhsFunction->getContext();

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
      auto memref =
          builder.create<memref::AllocOp>(UnknownLoc::get(ctx), memreftype);
      operands.push_back(memref);

      auto afterLastOperand = builder.saveInsertionPoint();

      SmallVector<mlir::Value> indices;
      auto argShape = arg.getType().cast<ShapedType>().getShape();
      for (auto dimSize : argShape) {
        // Create for op.
        auto forOp = builder.create<scf::ForOp>(
            UnknownLoc::get(ctx),
            builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), 0),
            builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx),
                                                   dimSize),
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
                                                     "__VERIFIER_nondet_float", operands);

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
      // Check for equality of lhsLoad and rhsLoad.
      auto cond1 = builder.create<arith::CmpFOp>(
          UnknownLoc::get(ctx), arith::CmpFPredicate::ONE, lhsLoad, rhsLoad);

      // Check for nan by calling function isnanf on lhsLoad.
      SmallVector<mlir::Value> cond2Operands = {lhsLoad};
      auto cond2 = builder.create<func::CallOp>(
          UnknownLoc::get(ctx), builder.getI1Type(), "!isnanf", cond2Operands);

      // Check for nan by calling function isnanf on rhsLoad.
      SmallVector<mlir::Value> cond3Operands = {rhsLoad};
      auto cond3 = builder.create<func::CallOp>(
          UnknownLoc::get(ctx), builder.getI1Type(), "!isnanf", cond3Operands);

      // Build predicate
      auto pred1 = builder.create<arith::AndIOp>(
          UnknownLoc::get(ctx), cond1.getResult(), cond2->getResult(0));
      auto pred2 = builder.create<arith::AndIOp>(
          UnknownLoc::get(ctx), pred1.getResult(), cond3->getResult(0));

      // Create if Operation.
      auto ifOp =
          builder.create<scf::IfOp>(UnknownLoc::get(ctx), pred2, /*withElseRegion=*/false);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

      SmallVector<mlir::Value> checkOperands = {};
      builder.create<func::CallOp>(UnknownLoc::get(ctx), builder.getF64Type(),
                                   "cbmc_assert", checkOperands);
    }
  }

  // Must have the check function declaration.
  builder.setInsertionPoint(&moduleBlock, moduleBlock.begin());
  func::FuncOp cbmcAssertFwdDecl = builder.create<func::FuncOp>(
      UnknownLoc::get(ctx), "cbmc_assert",
      mlir::FunctionType::get(ctx, {},
                              {builder.getF64Type()}));
  cbmcAssertFwdDecl.setPrivate();

  func::FuncOp isnanfFwdDecl = builder.create<func::FuncOp>(
      UnknownLoc::get(ctx), "!isnanf",
      mlir::FunctionType::get(ctx, {builder.getF64Type()},
                              {builder.getI1Type()}));
  isnanfFwdDecl.setPrivate();

  func::FuncOp cbmcDeclareFwdDecl = builder.create<func::FuncOp>(
      UnknownLoc::get(ctx), "__VERIFIER_nondet_float",
      mlir::FunctionType::get(ctx, {}, {builder.getF64Type()}));
  cbmcDeclareFwdDecl.setPrivate();

  // Must have a return op.
  builder.restoreInsertionPoint(afterLastCallOp);
  builder.create<func::ReturnOp>(UnknownLoc::get(ctx));

  return module;
}

bool checkValidate(func::FuncOp lhsFunction, func::FuncOp rhsFunction,
                   bool printArgsAndResults, bool printResults) {
  auto *ctx = lhsFunction->getContext();

  // Assemble module.
  auto module = buildModule(lhsFunction, rhsFunction);

  // Inline and lower to affine. The resulting IR should be in
  // SCF, MemRef, Arith and func dialect.
  auto pm = PassManager(ctx);
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createLowerAffinePass());
  pm.addPass(createMemrefCopyToLoopsPass());
  if (failed(pm.run(module.get()))) {
    llvm::errs() << "Could not inline or lower to affine\n";
    assert(false);
  }

  // Remove all functions except the main function.
  module->walk([](mlir::Operation *op) {
    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      if (funcOp.getName() != "main") {
        funcOp.erase();
      }
    }
  });

  // Translate the IR to C.
  if (failed(emitc::translateToCpp(module.get(), llvm::outs()))) {
    llvm::errs() << "Could not translate to Cpp with emitc\n";
    assert(false);
  }

  return true;
}
