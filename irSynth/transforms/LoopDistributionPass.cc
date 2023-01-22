#include "LoopDistributionPass.h"

#include "Utils.h"
#include "analysis/PolyhedralAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

void distributeLoops(Operation *op) {
  op->dump();

  Scop scop(op);
  auto dependenceGraph = scop.getDependenceGraph();
  dependenceGraph->computeDependencies();
  dependenceGraph->dump(llvm::outs());
}

void LoopDistributionPass::runOnOperation() {
  auto operation = getOperation();
  for (auto func : operation.getOps<func::FuncOp>()) {
    distributeLoops(func);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createLoopDistributionPass() {
  return std::make_unique<LoopDistributionPass>();
}
