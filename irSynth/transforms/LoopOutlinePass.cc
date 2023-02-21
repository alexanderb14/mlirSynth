#include "LoopOutlinePass.h"

#include "analysis/PolyhedralAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "transforms/Utils.h"

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

llvm::SetVector<Value> getOutOfBlockDefValues(mlir::Operation *op) {
  // Get all values.
  llvm::DenseMap<Value, bool> allValues;
  // - Defined in the block.
  op->walk([&](Operation *op) {
    for (auto result : op->getResults())
      allValues[result] = true;
  });
  // - Defined as arguments.
  for (unsigned i = 0; i < op->getNumRegions(); i++) {
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

  bool debug = false;
  if (debug)
    origFunc.dump();

  auto module = origFunc->getParentOfType<ModuleOp>();
  auto topLoops = getTopLevelLoops(origFunc);
  auto builder = OpBuilder::atBlockBegin(module.getBody());

  BlockAndValueMapping fnResultMapper;
  Operation *lastFunc = nullptr;
  Operation *lastCall = nullptr;

  unsigned loopCounter = 0;
  for (auto *topLoop : topLoops) {
    auto undefinedValues = getOutOfBlockDefValues(topLoop);
    auto loadedValues = getLoadedMemRefValues(topLoop);
    auto storedValues = getStoredMemRefValues(topLoop);

    if (debug) {
      llvm::outs() << "-----------------\n";
      topLoop->dump();
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
    func->setAttr("irsynth.original", builder.getUnitAttr());
    auto &bodyBlock = *func.addEntryBlock();

    // Add arguments to function.
    BlockAndValueMapping argMapper;

    // - Add loaded values as arguments.
    for (auto value : loadedValues) {
      if (undefinedValues.contains(value))
        continue;
      auto newArg = bodyBlock.addArgument(value.getType(), unknownLoc);
      argMapper.map(value, newArg);
    }
    // - Add undefined values as arguments or as local variables if they are
    // constants.
    for (auto value : undefinedValues) {
      if (argMapper.contains(value)) {
        continue;
      }

      // If the defining operation is a constant or a memref alloca, copy and
      // add it to the new function. Else, add it as an argument.
      auto *definingOp = value.getDefiningOp();
      // - Constant.
      if (definingOp && dyn_cast<arith::ConstantOp>(definingOp)) {
        auto constantOp = dyn_cast<arith::ConstantOp>(definingOp);
        auto newConstantOp = constantOp.clone();
        bodyBlock.push_back(newConstantOp);
        argMapper.map(value, newConstantOp.getResult());

      // - Memref alloca.
      } else if (definingOp && dyn_cast<memref::AllocaOp>(definingOp)) {
        auto allocaOp = dyn_cast<memref::AllocaOp>(definingOp);
        auto newAllocaOp = allocaOp.clone();
        bodyBlock.push_back(newAllocaOp);
        argMapper.map(value, newAllocaOp.getResult());

      // Else, add as argument.
      } else {
        auto newArg = bodyBlock.addArgument(value.getType(), unknownLoc);
        argMapper.map(value, newArg);
      }
    }
    auto reverseMapper = reverseMap(argMapper);

    // Add body.
    bodyBlock.push_back(topLoop->clone(argMapper));

    // Add the stored values as results.
    // As a heuristic, we use only the last-stored value as the result.
    // - Get the last stored value.
    llvm::SetVector<Value> storedValue;
    storedValue.insert(storedValues.back());

    // - Create return operation.
    llvm::SmallVector<Value> results;
    for (auto value : storedValue)
      results.push_back(argMapper.lookup(value));
    builder.setInsertionPoint(&bodyBlock, bodyBlock.end());
    builder.create<func::ReturnOp>(unknownLoc, results);

    // - Add the results to function type.
    llvm::SmallVector<Type> resultTypes;
    for (auto value : storedValue)
      resultTypes.push_back(value.getType());
    func.setFunctionType(
        builder.getFunctionType(bodyBlock.getArgumentTypes(), resultTypes));

    // - Add arg attributes of the original function.
    llvm::SmallVector<Attribute> argAttrs;
    if (auto origFuncArgAttrs = origFunc.getAllArgAttrs()) {
      for (auto arg : bodyBlock.getArguments()) {
        auto newArg = reverseMapper.lookup(arg);
        if (auto value = newArg.dyn_cast<BlockArgument>()) {
          auto attribute = origFuncArgAttrs[value.getArgNumber()];
          argAttrs.push_back(attribute);
        } else {
          newArg.dump();
          assert(false && "Unexpected value type");
        }
      }
      func.setAllArgAttrs(argAttrs);
    }

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
    builder.setInsertionPoint(topLoop);
    auto callOp = builder.create<func::CallOp>(unknownLoc, func.getSymName(),
                                               func.getResultTypes(), args);
    lastCall = callOp;

    // Add function call results to the fnResultMapper.
    for (unsigned i = 0; i < callOp.getNumResults(); i++)
      fnResultMapper.map(storedValue[i], callOp.getResult(i));

    // Remove the loop.
    topLoop->erase();
  }

  // Change the original function result to the result of the last outlined
  // function.
  // - Original function return type.
  auto lastFuncOp = dyn_cast<func::CallOp>(lastCall);
  auto lastFuncResultType = lastFuncOp.getResultTypes();
  origFunc.setFunctionType(
      builder.getFunctionType(origFunc.getArgumentTypes(), lastFuncResultType));

  // - Return value.
  auto *lastOp = origFunc.getBody().back().getTerminator();
  assert(isa<func::ReturnOp>(lastOp) && "Expected return op at the end of the "
                                        "function body.");
  auto returnOp = dyn_cast<func::ReturnOp>(lastOp);
  returnOp->setOperands(lastFuncOp->getResults());
}

void LoopOutlinePass::runOnOperation() {
  auto operation = getOperation();
  for (auto func : operation.getOps<func::FuncOp>())
    outlineLoops(func);
}

std::unique_ptr<OperationPass<ModuleOp>> createLoopOutlinePass() {
  return std::make_unique<LoopOutlinePass>();
}
