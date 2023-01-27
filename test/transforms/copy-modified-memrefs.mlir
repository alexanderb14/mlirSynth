// RUN: opt %s --copy-modified-memrefs | FileCheck %s

// CHECK:       func.func @foo(%arg0: memref<3x5xf64>, %arg1: memref<3x5xf64>, %arg2: memref<3x5xf64>) -> memref<3x5xf64> attributes {irsynth.original} {
// CHECK-NEXT:    %alloc = memref.alloc() : memref<3x5xf64>
// CHECK-NEXT:    memref.copy %arg2, %alloc : memref<3x5xf64> to memref<3x5xf64>
// CHECK-NEXT:    affine.for %arg3 = 0 to 3 {
// CHECK-NEXT:      affine.for %arg4 = 0 to 5 {
// CHECK-NEXT:        %0 = affine.load %arg0[%arg3, %arg4] : memref<3x5xf64>
// CHECK-NEXT:        %1 = affine.load %arg1[%arg3, %arg4] : memref<3x5xf64>
// CHECK-NEXT:        %2 = arith.addf %0, %1 : f64
// CHECK-NEXT:        affine.store %2, %alloc[%arg3, %arg4] : memref<3x5xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return %alloc : memref<3x5xf64>
// CHECK-NEXT:  }
module {
  func.func @foo(%arg0: memref<3x5xf64>, %arg1: memref<3x5xf64>, %res: memref<3x5xf64>) -> memref<3x5xf64> attributes {irsynth.original} {
    affine.for %i = 0 to 3 {
      affine.for %j = 0 to 5 {
        %0 = affine.load %arg0[%i, %j] : memref<3x5xf64>
        %1 = affine.load %arg1[%i, %j] : memref<3x5xf64>
        %2 = arith.addf %0, %1 : f64
        affine.store %2, %res[%i, %j] : memref<3x5xf64>
      }
    }

    return %res : memref<3x5xf64>
  }
}
