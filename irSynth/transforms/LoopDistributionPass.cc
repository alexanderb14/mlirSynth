#include "LoopDistributionPass.h"

#include "analysis/PolyhedralAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

void distributeLoops(func::FuncOp &op) {
  op.dump();

  Scop scop(op);
  auto dependenceGraph = scop.getDependenceGraph();
  dependenceGraph->computeDependencies();
  dependenceGraph->dump(llvm::outs());
}

struct LoopDistributionPass
    : public PassWrapper<LoopDistributionPass, OperationPass<func::FuncOp>> {

  void runOnOperation() override {
    func::FuncOp op = getOperation();

    distributeLoops(op);
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createLoopDistributionPass() {
  return std::make_unique<LoopDistributionPass>();
}
