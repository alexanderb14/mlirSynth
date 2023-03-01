#ifndef IRSYNTH_TRANSFORMS_UTILS_H
#define IRSYNTH_TRANSFORMS_UTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

llvm::SmallVector<mlir::Operation *> getTopLevelLoops(mlir::func::FuncOp &op);

llvm::SmallVector<mlir::Value> getOutOfBlockDefValues(mlir::Block *block);
llvm::SetVector<mlir::Value> getLoadedMemRefValues(mlir::Operation *op);
llvm::SetVector<mlir::Value> getStoredMemRefValues(mlir::Operation *op);

#endif // IRSYNTH_TRANSFORMS_UTILS_H
