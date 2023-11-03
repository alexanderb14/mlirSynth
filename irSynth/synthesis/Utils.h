#ifndef IRSYNTH_UTILS_H
#define IRSYNTH_UTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

mlir::OwningOpRef<mlir::ModuleOp> createModule(mlir::MLIRContext &ctx,
                                               mlir::func::FuncOp *function);
mlir::OwningOpRef<mlir::ModuleOp> createModule(mlir::MLIRContext &ctx,
                                               mlir::Region *region);

#endif // IRSYNTH_UTILS_H
