#include "DeadArgRemovalPass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

struct DeadArgRemovalPass
    : public PassWrapper<DeadArgRemovalPass, OperationPass<func::FuncOp>> {

  void runOnOperation() override { func::FuncOp op = getOperation(); }
};

std::unique_ptr<OperationPass<func::FuncOp>> createDeadArgRemovalPass() {
  return std::make_unique<DeadArgRemovalPass>();
}
