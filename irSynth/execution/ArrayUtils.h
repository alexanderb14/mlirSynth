#ifndef TOOLS_SYNTHESIZER_ARRAYUTILS_H
#define TOOLS_SYNTHESIZER_ARRAYUTILS_H

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>

void printArray(double *arr, llvm::ArrayRef<int64_t> shape);
double hashArray(double *arr, llvm::ArrayRef<int64_t> shape);
bool areArraysEqual(double *arr1, double *arr2, llvm::ArrayRef<int64_t> shape);

#endif // TOOLS_SYNTHESIZER_ARRAYUTILS_H
