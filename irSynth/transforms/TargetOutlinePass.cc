#include "TargetOutlinePass.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "transforms/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Location.h"

using namespace mlir;

llvm::SmallVector<llvm::SmallVector<Operation *>>
detectMaximumChains(func::FuncOp func, std::string &targetDialect) {
  bool debug = false;
  if (debug) {
    llvm::outs() << "Dialects of operations in function " << func.getName()
                 << ":\n";
    for (auto &op : func.getBlocks().front()) {
      llvm::outs() << op.getDialect()->getNamespace() << "\n";
    }
    llvm::outs() << "------------\n";
  }

  // Find the maximum of subsequent use-def chained operations of a given target
  // dialect with a linear scan.
  llvm::SmallVector<llvm::SmallVector<Operation *>> chains;

  llvm::SmallVector<Operation *> currentChain;
  llvm::DenseMap<Operation *, bool> seen;

  // BFS. Init worklist with all operations in the function.
  llvm::SmallVector<Operation *> worklist;
  for (auto &op : func.getBlocks().front()) {
    worklist.push_back(&op);
  }

  while (!worklist.empty()) {
    auto *op = worklist.pop_back_val();

    // Skip if we've already seen this operation.
    if (seen.count(op)) {
      continue;
    }
    seen[op] = true;

    // If the operation is in the target dialect, add it to the current chain.
    if (op->getDialect()->getNamespace() == targetDialect) {
      currentChain.push_back(op);
    }

    // If the operation is not in the target dialect, and the current chain is
    // non-empty, add the current chain to the list of chains and reset the
    // current chain.
    if (op->getDialect()->getNamespace() != targetDialect &&
        !currentChain.empty()) {
      chains.push_back(currentChain);
      currentChain.clear();
    }

    // Add all of the operation's users to the worklist.
    for (auto *user : op->getUsers()) {
      worklist.push_back(user);
    }
  }

  // If the current chain is non-empty, add it to the list of chains.
  if (!currentChain.empty()) {
    chains.push_back(currentChain);
  }

  // Reverse each chain so that the operations are in the order they appear in
  // the function.
  for (auto &chain : chains) {
    std::reverse(chain.begin(), chain.end());
  }

  if (debug) {
    llvm::outs() << "Chains:\n";
    for (auto &chain : chains) {
      for (auto *op : chain) {
        op->print(llvm::outs());
        llvm::outs() << "\n";
      }
      llvm::outs() << "----\n";
    }
  }

  return chains;
}

llvm::SmallVector<Value> createFunction(OpBuilder &builder, std::string functionName, SmallVector<Operation *> &chain) {
  auto unknownLoc = UnknownLoc::get(builder.getContext());

  // Create a new function.
  auto func = builder.create<func::FuncOp>(
      unknownLoc, functionName,
      builder.getFunctionType({}, {}));
  func->setAttr("irsynth.target", builder.getUnitAttr());
  auto &bodyBlock = *func.addEntryBlock();
  builder.setInsertionPoint(&bodyBlock, bodyBlock.begin());

  // - Move the operations defined in the chain into the new function.
  for (auto *op : chain) {
    op->moveBefore(&bodyBlock, bodyBlock.end());
  }

  // - Add arguments to the new function.
  BlockAndValueMapping argMapper;

  auto undefinedValues = getOutOfBlockDefValues(&bodyBlock);

  for (auto value : undefinedValues) {
    auto newArg = bodyBlock.addArgument(value.getType(), unknownLoc);
    argMapper.map(value, newArg);
  }

  // - Remap the uses of undefined values to the new arguments.
  for (auto value : undefinedValues) {
    for (auto *user : value.getUsers()) {
      user->replaceUsesOfWith(value, argMapper.lookup(value));
    }
  }

  // - Add return.
  auto returnOp =
      builder.create<func::ReturnOp>(unknownLoc);
  returnOp->setOperands(chain.back()->getResults());
  func.setFunctionType(
      builder.getFunctionType(bodyBlock.getArgumentTypes(), {chain.back()->getResult(0).getType()}));

  return undefinedValues;
}

void outlineTargets(ModuleOp &module, func::FuncOp &origFunc, std::string targetDialect) {
  auto unknownLoc = UnknownLoc::get(module.getContext());

  // Detect maximum chains of consecutive operations in the target dialect.
  auto chains = detectMaximumChains(origFunc, targetDialect);

  OpBuilder builder(origFunc.getContext());
  builder.setInsertionPointToStart(module.getBody());

  unsigned functionId = 0;
  for (auto &chain : chains) {
    auto *callInsertionPoint = chain.back()->getNextNode();
    auto oldUsers = chain.back()->getUsers();

    // Create a new function with the operations in the chain moved into it.
    std::string functionName = "target_" + std::to_string(functionId++);
    auto args = createFunction(builder, functionName, chain);

    // Add a call to the new function.
    llvm::SmallVector<Type> resultTypes;
    for (auto result : chain.back()->getResults()) {
      resultTypes.push_back(result.getType());
    }

    builder.setInsertionPoint(callInsertionPoint);
    auto callOp = builder.create<func::CallOp>(
       unknownLoc, functionName, resultTypes, args);

    BlockAndValueMapping fnResultMapper;
    for (auto &result : llvm::enumerate(chain.back()->getResults())) {
      fnResultMapper.map(result.value(), callOp.getResult(result.index()));
    }

    // Replace the old users operands with the results of the call.
    for (auto *user : oldUsers) {
      for (auto &operand : llvm::enumerate(user->getOperands())) {
        if (operand.value() == chain.back()->getResult(0)) {
          user->setOperand(operand.index(), callOp.getResult(0));
        }
      }
    }
  }
}

void TargetOutlinePass::runOnOperation() {
  auto operation = getOperation();
  for (auto func : operation.getOps<func::FuncOp>())
    outlineTargets(operation, func, "stablehlo");
}

std::unique_ptr<OperationPass<ModuleOp>> createTargetOutlinePass() {
  return std::make_unique<TargetOutlinePass>();
}
