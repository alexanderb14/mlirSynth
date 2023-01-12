#ifndef IRSYNTH_ARGUTILS_H
#define IRSYNTH_ARGUTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"

using OwningMemRef0DPtr = mlir::OwningMemRef<double, 0> *;
using OwningMemRef1DPtr = mlir::OwningMemRef<double, 1> *;
using OwningMemRef2DPtr = mlir::OwningMemRef<double, 2> *;
using OwningMemRef3DPtr = mlir::OwningMemRef<double, 3> *;

using Result0DPtr = mlir::ExecutionEngine::Result<OwningMemRef0DPtr> *;
using Result1DPtr = mlir::ExecutionEngine::Result<OwningMemRef1DPtr> *;
using Result2DPtr = mlir::ExecutionEngine::Result<OwningMemRef2DPtr> *;
using Result3DPtr = mlir::ExecutionEngine::Result<OwningMemRef3DPtr> *;

using DoublePtr = double *;
using ReturnAndArgType =
    std::variant<OwningMemRef0DPtr, OwningMemRef1DPtr, OwningMemRef2DPtr,
                 OwningMemRef3DPtr, Result0DPtr, Result1DPtr, Result2DPtr,
                 Result3DPtr, DoublePtr>;

ReturnAndArgType getOwningMemRefForShape(mlir::ArrayRef<int64_t> shape);
ReturnAndArgType getReturnMemRefForShape(mlir::ArrayRef<int64_t> shape);
llvm::ArrayRef<int64_t> getReturnShape(mlir::func::FuncOp function);
void addReturn(mlir::ArrayRef<int64_t> returnShape,
               std::vector<ReturnAndArgType> &returnAndArgs);
std::vector<ReturnAndArgType> createArgs(mlir::Region::BlockArgListType args);
double *getReturnDataPtr(ReturnAndArgType &returnAndArgs);
void randomlyInitializeArgs(std::vector<ReturnAndArgType> args);
void printArgs(std::vector<ReturnAndArgType> args);
std::vector<ReturnAndArgType>
selectArgs(const std::vector<ReturnAndArgType> &args,
           const std::vector<unsigned> &argIds);
void printArgTypes(std::vector<ReturnAndArgType> args);
void createReturnAndArgsArray(std::vector<ReturnAndArgType> returnAndArgs,
                              llvm::SmallVector<void *> &returnAndArgsPtrs,
                              llvm::SmallVector<void *> &returnAndArgsPtrsPtrs);
void convertScalarToMemrefArgs(std::vector<ReturnAndArgType> &returnAndArgs);

#endif // IRSYNTH_ARGUTILS_H
