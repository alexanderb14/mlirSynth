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

bool debug = false;

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

void outlineLoops(func::FuncOp &origFunc) {
  auto unknownLoc = UnknownLoc::get(origFunc.getContext());

  if (debug)
    origFunc.dump();

  auto module = origFunc->getParentOfType<ModuleOp>();
  auto loops = getTopLevelLoops(origFunc);
  auto builder = OpBuilder::atBlockBegin(module.getBody());

  BlockAndValueMapping fnResultMapper;
  Operation *lastFunc = nullptr;

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
    OpBuilder builder(origFunc.getContext());
    auto func = builder.create<func::FuncOp>(
        unknownLoc, "fn_" + std::to_string(loopCounter++),
        builder.getFunctionType({}, {}));
    func->setAttr("irsynth.synthesize", builder.getUnitAttr());
    auto &bodyBlock = *func.addEntryBlock();

    // Add arguments to function.
    BlockAndValueMapping argMapper;

    // - Add loaded values as arguments.
    for (auto value : loadedValues) {
      auto newArg = bodyBlock.addArgument(value.getType(), unknownLoc);
      argMapper.map(value, newArg);
    }
    // - Add undefined values as arguments or as local variables if they are
    // constants.
    for (auto value : undefinedValues) {
      if (argMapper.contains(value))
        continue;

      // If the defining operation is a constant, copy and add it to the new
      // function. Else, add it as an argument.
      auto *definingOp = value.getDefiningOp();
      if (definingOp && dyn_cast<arith::ConstantOp>(definingOp)) {
        auto constantOp = dyn_cast<arith::ConstantOp>(definingOp);
        auto newConstantOp = constantOp.clone();
        bodyBlock.push_back(newConstantOp);
        argMapper.map(value, newConstantOp.getResult());
      } else {
        auto newArg = bodyBlock.addArgument(value.getType(), unknownLoc);
        argMapper.map(value, newArg);
      }
    }

    // Add body.
    bodyBlock.push_back(loop->clone(argMapper));

    // Add the stored values as results.
    // - Create return operation.
    llvm::SmallVector<Value> results;
    for (auto value : storedValues)
      results.push_back(argMapper.lookup(value));
    builder.setInsertionPoint(&bodyBlock, bodyBlock.end());
    auto returnOp = builder.create<func::ReturnOp>(unknownLoc, results);

    // - Add the results to function type.
    llvm::SmallVector<Type> resultTypes;
    for (auto value : storedValues)
      resultTypes.push_back(value.getType());
    func.setFunctionType(
        builder.getFunctionType(bodyBlock.getArgumentTypes(), resultTypes));

    // Insert the new function and replace the loop with a call to it.
    // ---------------------------------------------
    if (!lastFunc)
      builder.setInsertionPointToStart(module.getBody());
    else
      builder.setInsertionPointAfter(lastFunc);

    builder.insert(func);
    lastFunc = func;

    // Create args for the call.
    llvm::SmallVector<Value> args;
    auto reverseMapper = reverseMap(argMapper);
    for (auto arg : func.getArguments()) {
      auto value = reverseMapper.lookupOrNull(arg);

      // If the arg value was already recomputed by an earlier call, use this
      // one.
      if (fnResultMapper.contains(value))
        args.push_back(fnResultMapper.lookup(value));
      else
        args.push_back(value);
    }

    // Create function call.
    builder.setInsertionPoint(loop);
    auto callOp = builder.create<func::CallOp>(unknownLoc, func.getSymName(),
                                               func.getResultTypes(), args);

    // Add function call results to the fnResultMapper.
    for (int i = 0; i < callOp.getNumResults(); i++)
      fnResultMapper.map(storedValues[i], callOp.getResult(i));

    // Remove the loop.
    loop->erase();
  }
}

struct LoopOutlinePass
    : public PassWrapper<LoopOutlinePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    auto operation = getOperation();
    for (auto func : operation.getOps<func::FuncOp>())
      outlineLoops(func);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createLoopOutlinePass() {
  return std::make_unique<LoopOutlinePass>();
}
