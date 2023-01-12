#import "ArrayUtils.h"

#include "llvm/Support/raw_ostream.h"

#include <math.h>

using namespace llvm;

void printArray(double *arr, ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    llvm::outs() << arr[0] << "\n";
  } else if (shape.size() == 1) {
    llvm::outs() << "[";
    for (int i = 0; i < shape[0]; i++) {
      llvm::outs() << arr[i];
      if (i < shape[0] - 1)
        llvm::outs() << ", ";
    }
    llvm::outs() << "]\n";
  } else if (shape.size() == 2) {
    llvm::outs() << "[";
    for (int i = 0; i < shape[0]; i++) {
      llvm::outs() << "[";
      for (int j = 0; j < shape[1]; j++) {
        llvm::outs() << arr[i * shape[1] + j];
        if (j < shape[1] - 1)
          llvm::outs() << ", ";
      }
      llvm::outs() << "]";
      if (i < shape[0] - 1)
        llvm::outs() << ",\n";
    }
    llvm::outs() << "]\n";
  } else if (shape.size() == 3) {
    llvm::outs() << "[";
    for (int i = 0; i < shape[0]; i++) {
      llvm::outs() << "[";
      for (int j = 0; j < shape[1]; j++) {
        llvm::outs() << "[";
        for (int k = 0; k < shape[2]; k++) {
          llvm::outs() << arr[i * shape[1] * shape[2] + j * shape[2] + k];
          if (k < shape[2] - 1)
            llvm::outs() << ", ";
        }
        llvm::outs() << "]";
        if (j < shape[1] - 1)
          llvm::outs() << ",\n";
      }
      llvm::outs() << "]";
      if (i < shape[0] - 1)
        llvm::outs() << ",\n";
    }
    llvm::outs() << "]\n";
  } else {
    assert(false && "Unsupported shape");
  }
}

double hashArray(double *arr, ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    return *arr;
  }
  if (shape.size() == 1) {
    double sum = 0;
    for (int i = 0; i < shape[0]; i++) {
      sum += arr[i];
    }
    return sum / shape[0] * 7.331;
  }
  if (shape.size() == 2) {
    double sum = 0;
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        sum += arr[i * shape[1] + j];
      }
    }
    return sum / (shape[0] * 1.337 + shape[1] * 0.337);
  }
  if (shape.size() == 3) {
    double sum = 0;
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        for (int k = 0; k < shape[2]; k++) {
          sum += arr[i * shape[1] * shape[2] + j * shape[2] + k];
        }
      }
    }
    return sum / (shape[0] * 1.337 + shape[1] * 0.337);
  }
  assert(false && "Unsupported shape");
}

bool areArraysEqual(double *arr1, double *arr2, ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    return (floor(*arr1 * 1000) != floor(*arr2 * 1000));
  }
  if (shape.size() == 1) {
    for (int i = 0; i < shape[0]; i++) {
      if (floor(arr1[i] * 1000) != floor(arr2[i] * 1000)) {
        return false;
      }
    }
    return true;
  }
  if (shape.size() == 2) {
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        if (floor(arr1[i * shape[1] + j] * 1000) !=
            floor(arr2[i * shape[1] + j] * 1000)) {
          return false;
        }
      }
    }
    return true;
  }
  if (shape.size() == 3) {
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        for (int k = 0; k < shape[2]; k++) {
          if (floor(arr1[i * shape[1] * shape[2] + j * shape[2] + k] * 1000) !=
              floor(arr2[i * shape[1] * shape[2] + j * shape[2] + k] * 1000)) {
            return false;
          }
        }
      }
    }
    return true;
  }
  assert(false && "Unsupported shape");
}
