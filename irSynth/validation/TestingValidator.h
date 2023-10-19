#ifndef IRSYNTH_VALIDATION_TESTINGVALIDATOR_H
#define IRSYNTH_VALIDATION_TESTINGVALIDATOR_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

bool testValidate(mlir::func::FuncOp lhsFunction,
                  mlir::func::FuncOp rhsFunction,
                  bool printArgsAndResults = false, bool printResults = false);

#endif // IRSYNTH_VALIDATION_TESTINGVALIDATOR_H
