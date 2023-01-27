#ifndef IRSYNTH_ENUMERATION_UTILS_H
#define IRSYNTH_ENUMERATION_UTILS_H

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"

std::vector<mlir::StringAttr>

getFilteredAttributeNames(mlir::RegisteredOperationName opName);

int getRequiredNumOperands(mlir::Operation *op);
int getRequiredNumAttributes(mlir::Operation *op);
int getRequiredNumRegions(mlir::Operation *op);
int getRequiredNumResults(mlir::Operation *op);

std::tuple<int, int>
getRequiredNumOperandsAndNumResults(mlir::MLIRContext &ctx,
                                    mlir::OperationName &opName);

#endif // IRSYNTH_ENUMERATION_UTILS_H
