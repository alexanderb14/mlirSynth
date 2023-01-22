#ifndef IRSYNTH_UTILS_H
#define IRSYNTH_UTILS_H

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"

#include <random>

int randomInteger(int min, int max);

llvm::SmallVector<mlir::Operation *> getTopLevelLoops(mlir::func::FuncOp &op);

std::vector<std::shared_ptr<mlir::Region>> getRegions(mlir::OpBuilder &builder);

mlir::Operation *createDummyOperation(mlir::MLIRContext &ctx,
                                      mlir::OperationName &opName);

int getRequiredNumOperands(mlir::Operation *op);

std::vector<mlir::StringAttr>
getFilteredAttributeNames(mlir::RegisteredOperationName opName);
int getRequiredNumAttributes(mlir::Operation *op);

int getRequiredNumRegions(mlir::Operation *op);

int getRequiredNumResults(mlir::Operation *op);

std::tuple<int, int>
getRequiredNumOperandsAndNumResults(mlir::MLIRContext &ctx,
                                    mlir::OperationName &opName);

#endif // IRSYNTH_UTILS_H
