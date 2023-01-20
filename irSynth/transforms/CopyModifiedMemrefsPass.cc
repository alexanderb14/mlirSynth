#include "CopyModifiedMemrefsPass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

void copyModifiedMemrefs(func::FuncOp &op) {
}

struct CopyModifiedMemrefsPass
    : public PassWrapper<CopyModifiedMemrefsPass, OperationPass<func::FuncOp>> {

  void runOnOperation() override {
    func::FuncOp op = getOperation();

    copyModifiedMemrefs(op);
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createCopyModifiedMemrefsPass() {
  return std::make_unique<CopyModifiedMemrefsPass>();
}
