#ifndef IRSYNTH_VALIDATION_CHECKINGVALIDATOR_H
#define IRSYNTH_VALIDATION_CHECKINGVALIDATOR_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

bool checkValidate(mlir::func::FuncOp lhsFunction,
                   mlir::func::FuncOp rhsFunction,
                   bool printArgsAndResults = false, bool printResults = false);

#endif // IRSYNTH_VALIDATION_CHECKINGVALIDATOR_H
