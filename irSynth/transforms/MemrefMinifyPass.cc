#include "MemrefMinifyPass.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

void minifyMemrefs(ModuleOp &op) {
}

struct MemrefMinifyPass
    : public PassWrapper<MemrefMinifyPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    auto operation = getOperation();
    minifyMemrefs(operation);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createMemrefMinifyPass() {
  return std::make_unique<MemrefMinifyPass>();
}
