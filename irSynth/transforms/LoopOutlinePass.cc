#include "LoopOutlinePass.h"

#include "analysis/PolyhedralAnalysis.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

llvm::SmallVector<mlir::Operation *> getTopLevelLoops(func::FuncOp &op) {
  llvm::SmallVector<mlir::Operation *> loops;
  assert(op.getBody().getBlocks().size() == 1);
  auto &block = op.getBody().getBlocks().front();
  for (auto &op : block.getOperations()) {
    if (dyn_cast<AffineForOp>(op)) {
      loops.push_back(&op);
    }
  }
  return loops;
}

llvm::SetVector<Value> getOutOfBlockDefValues(mlir::Operation *op) {
  llvm::SetVector<Value> undefinedValues;
  op->walk([&](mlir::Operation *subOp) {
    for (auto operand : subOp->getOperands()) {
      if (operand.getDefiningOp() &&
          operand.getDefiningOp()->getBlock() == op->getBlock()) {
        undefinedValues.insert(operand);
      }
    }
  });
  return undefinedValues;
}

llvm::SetVector<Value> getLoadedMemRefValues(mlir::Operation *op) {
  llvm::SetVector<Value> values;
  op->walk([&](AffineLoadOp loadOp) { values.insert(loadOp.getMemRef()); });
  return values;
}

llvm::SetVector<Value> getStoredMemRefValues(mlir::Operation *op) {
  llvm::SetVector<Value> values;
  op->walk([&](AffineStoreOp storeOp) { values.insert(storeOp.getMemRef()); });
  return values;
}

void outlineLoops(func::FuncOp &op) {
  bool debug = false;
  if (debug)
    op.dump();

  auto loops = getTopLevelLoops(op);
  for (auto *loop : loops) {

    auto undefinedValues = getOutOfBlockDefValues(loop);
    auto loadedValues = getLoadedMemRefValues(loop);
    auto storedValues = getStoredMemRefValues(loop);

    if (debug) {
      llvm::outs() << "--------------------------------\n";
      loop->dump();

      llvm::outs() << "Undefined values:\n";
      for (auto value : undefinedValues) {
        value.dump();
      }

      llvm::outs() << "Loaded:\n";
      for (auto value : loadedValues) {
        value.dump();
      }

      llvm::outs() << "Stored:\n";
      for (auto value : storedValues) {
        value.dump();
      }
    }

    // Create a new function.
    auto unknownLoc = UnknownLoc::get(op.getContext());
    OpBuilder builder(op.getContext());
    auto func = builder.create<func::FuncOp>(unknownLoc, "foo",
                                             builder.getFunctionType({}, {}));
    auto &bodyBlock = *func.addEntryBlock();

    BlockAndValueMapping mapper;

    // Add the undefined values as arguments or operations.
    for (auto value : undefinedValues) {
      auto *definingOp = value.getDefiningOp();

      // If the defining operation is a constant, copy and add it to the new
      // function.
      if (auto constantOp = dyn_cast<arith::ConstantOp>(definingOp)) {
        auto newConstantOp = constantOp.clone();
        bodyBlock.push_back(newConstantOp);
        mapper.map(value, newConstantOp.getResult());
      }

      // All the remaining values are added as arguments.
      else {
        auto newArg = bodyBlock.addArgument(value.getType(), unknownLoc);
        mapper.map(value, newArg);
      }
    }

    // Add the loaded values as arguments.
    for (auto value : loadedValues) {
      auto newArg = bodyBlock.addArgument(value.getType(), unknownLoc);
      mapper.map(value, newArg);
    }

    // Add the loop to the body block.
    bodyBlock.push_back(loop->clone(mapper));

    // Add the stored values as results.
    // - Create return operation.
    llvm::SmallVector<Value> results;
    for (auto value : storedValues)
      results.push_back(mapper.lookup(value));
    builder.setInsertionPoint(&bodyBlock, bodyBlock.end());
    auto returnOp = builder.create<func::ReturnOp>(unknownLoc, results);

    // - Add the results to function type.
    llvm::SmallVector<Type> resultTypes;
    for (auto value : storedValues)
      resultTypes.push_back(value.getType());
    func.setType(
        builder.getFunctionType(bodyBlock.getArgumentTypes(), resultTypes));

    func.dump();
  }
}

struct LoopOutlinePass
    : public PassWrapper<LoopOutlinePass, OperationPass<func::FuncOp>> {

  void runOnOperation() override {
    func::FuncOp op = getOperation();

    outlineLoops(op);
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createLoopOutlinePass() {
  return std::make_unique<LoopOutlinePass>();
}
