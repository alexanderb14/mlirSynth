#ifndef IRSYNTH_TRANSFORMS_UTILS_H
#define IRSYNTH_TRANSFORMS_UTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

llvm::SmallVector<mlir::Operation *> getTopLevelLoops(mlir::func::FuncOp &op);

#endif // IRSYNTH_TRANSFORMS_UTILS_H
