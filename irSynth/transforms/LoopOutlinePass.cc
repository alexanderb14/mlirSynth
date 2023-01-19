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
  // Get all values.
  llvm::DenseMap<Value, bool> allValues;
  // - Defined in the block.
  op->walk([&](Operation *op) {
    for (auto result : op->getResults())
      allValues[result] = true;
  });
  // - Defined as arguments.
  for (int i = 0; i < op->getNumRegions(); i++) {
    for (auto &block : op->getRegion(i).getBlocks()) {
      for (auto arg : block.getArguments())
        allValues[arg] = true;
    }
  }

  // Get all ops.
  llvm::SetVector<Operation *> allOps;
  op->walk([&](Operation *op) { allOps.insert(op); });

  llvm::SetVector<Value> undefinedValues;
  for (auto &op : allOps) {
    for (auto operand : op->getOperands()) {
      if (allValues.count(operand) == 0 && !operand.getType().isa<IndexType>())
        undefinedValues.insert(operand);
    }
  }

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

BlockAndValueMapping reverseMap(BlockAndValueMapping &mapper) {
  BlockAndValueMapping reverseMapper;
  for (auto &pair : mapper.getValueMap())
    reverseMapper.map(pair.second, pair.first);
  return reverseMapper;
}

void outlineLoops(func::FuncOp &op) {
  auto unknownLoc = UnknownLoc::get(op.getContext());

  bool debug = true;
  if (debug)
    op.dump();

  auto module = op->getParentOfType<ModuleOp>();
  auto loops = getTopLevelLoops(op);
  auto builder = OpBuilder::atBlockBegin(module.getBody());

  unsigned loopCounter = 0;
  for (auto *loop : loops) {
    auto undefinedValues = getOutOfBlockDefValues(loop);
    auto loadedValues = getLoadedMemRefValues(loop);
    auto storedValues = getStoredMemRefValues(loop);

    if (debug) {
      llvm::outs() << "-----------------\n";
      loop->dump();
      llvm::outs() << "Undefined values:\n";
      for (auto value : undefinedValues)
        value.dump();
      llvm::outs() << "Loaded:\n";
      for (auto value : loadedValues)
        value.dump();
      llvm::outs() << "Stored:\n";
      for (auto value : storedValues)
        value.dump();
    }

    // Create a new function.
    // ---------------------------------------------
    OpBuilder builder(op.getContext());
    auto func = builder.create<func::FuncOp>(
        unknownLoc, "fn_" + std::to_string(loopCounter++),
        builder.getFunctionType({}, {}));
    auto &bodyBlock = *func.addEntryBlock();

    // Add arguments to function.
    BlockAndValueMapping mapper;

    // - Add loaded values as arguments.
    for (auto value : loadedValues) {
      auto newArg = bodyBlock.addArgument(value.getType(), unknownLoc);
      mapper.map(value, newArg);
    }
    // - Add undefined values as arguments or as local variables if they are
    // constants.
    for (auto value : undefinedValues) {
      if (mapper.contains(value))
        continue;

      // If the defining operation is a constant, copy and add it to the new
      // function. Else, add it as an argument.
      auto *definingOp = value.getDefiningOp();
      if (definingOp && dyn_cast<arith::ConstantOp>(definingOp)) {
        auto constantOp = dyn_cast<arith::ConstantOp>(definingOp);
        auto newConstantOp = constantOp.clone();
        bodyBlock.push_back(newConstantOp);
        mapper.map(value, newConstantOp.getResult());
      } else {
        auto newArg = bodyBlock.addArgument(value.getType(), unknownLoc);
        mapper.map(value, newArg);
      }
    }

    // - Add the stored values as last arguments.
    for (auto value : storedValues) {
      if (mapper.contains(value))
        continue;
      auto newArg = bodyBlock.addArgument(value.getType(), unknownLoc);
      mapper.map(value, newArg);
    }

    // Add body.
    bodyBlock.push_back(loop->clone(mapper));

    // Add return.
    llvm::SmallVector<Value> results;
    builder.setInsertionPoint(&bodyBlock, bodyBlock.end());
    auto returnOp = builder.create<func::ReturnOp>(unknownLoc, results);

    // Add function type, as it has been created without any earlier.
    func.setFunctionType(
        builder.getFunctionType(bodyBlock.getArgumentTypes(), {}));
    func.dump();

    // Insert the new function and replace the loop with a call to it.
    // ---------------------------------------------
    builder.setInsertionPointToStart(module.getBody());
    builder.insert(func);

    llvm::SmallVector<Value> args;
    auto reverseMapper = reverseMap(mapper);
    for (auto value : func.getArguments())
      args.push_back(reverseMapper.lookupOrNull(value));

    // Create function call.
    builder.setInsertionPoint(loop);
    auto callOp = builder.create<func::CallOp>(unknownLoc, func.getSymName(),
                                               func->getResultTypes(), args);

    // Remove the loop.
    loop->erase();
  }

  module.dump();
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
