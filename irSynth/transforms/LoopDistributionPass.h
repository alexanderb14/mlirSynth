#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createLoopDistributionPass();
