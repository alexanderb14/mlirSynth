#ifndef IRSYNTH_ARRAYUTILS_H
#define IRSYNTH_ARRAYUTILS_H

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <mutex>

void printArray(double *arr, llvm::ArrayRef<int64_t> shape, std::mutex &printMutex);
double hashArray(double *arr, llvm::ArrayRef<int64_t> shape);
bool areArraysEqual(double *arr1, double *arr2, llvm::ArrayRef<int64_t> shape);

#endif // IRSYNTH_ARRAYUTILS_H
