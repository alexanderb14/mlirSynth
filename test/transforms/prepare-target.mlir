// RUN: opt %s --prepare-target | FileCheck %s

// CHECK:       func.func @main(%arg0: memref<3x5xf64>) -> memref<3x5xf64> {
// CHECK-NEXT:    affine.for %arg1 = 0 to 3 {
// CHECK-NEXT:      affine.for %arg2 = 0 to 5 {
// CHECK-NEXT:        %0 = affine.load %arg0[%arg1, %arg2] : memref<3x5xf64>
// CHECK-NEXT:        %1 = arith.mulf %0, %0 : f64
// CHECK-NEXT:        affine.store %1, %arg0[%arg1, %arg2] : memref<3x5xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return %arg0 : memref<3x5xf64>
// CHECK-NEXT:  }
module {
  func.func @fn_0(%arg0: memref<3x5xf64>, %arg1: memref<3x5xf64>, %arg2: memref<3x5xf64>) -> memref<3x5xf64> attributes {irsynth.original} {
    affine.for %arg3 = 0 to 3 {
      affine.for %arg4 = 0 to 5 {
        %0 = affine.load %arg0[%arg3, %arg4] : memref<3x5xf64>
        %1 = affine.load %arg1[%arg3, %arg4] : memref<3x5xf64>
        %2 = arith.addf %0, %1 : f64
        affine.store %2, %arg2[%arg3, %arg4] : memref<3x5xf64>
      }
    }
    return %arg2 : memref<3x5xf64>
  }
  func.func @fn_1(%arg0: memref<3x5xf64>) -> memref<3x5xf64> attributes {irsynth.target} {
    affine.for %arg1 = 0 to 3 {
      affine.for %arg2 = 0 to 5 {
        %0 = affine.load %arg0[%arg1, %arg2] : memref<3x5xf64>
        %1 = arith.mulf %0, %0 : f64
        affine.store %1, %arg0[%arg1, %arg2] : memref<3x5xf64>
      }
    }
    return %arg0 : memref<3x5xf64>
  }
}
