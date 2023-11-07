// RUN: opt %s --outline-loops | FileCheck %s

// CHECK:       func.func @fn_0(%
// CHECK:       func.func @fn_1(%

// CHECK:         %0 = call @fn_0(%arg0, %arg1, %alloc) : (memref<3x5xf64>, memref<3x5xf64>, memref<3x5xf64>) -> memref<3x5xf64>
// CHECK-NEXT:    %1 = call @fn_1(%0) : (memref<3x5xf64>) -> memref<3x5xf64>
module {
  func.func @foo(%arg0: memref<3x5xf64>, %arg1: memref<3x5xf64>) -> memref<3x5xf64> {
    %res = memref.alloc() : memref<3x5xf64>

    affine.for %i = 0 to 3 {
      affine.for %j = 0 to 5 {
        %0 = affine.load %arg0[%i, %j] : memref<3x5xf64>
        %1 = affine.load %arg1[%i, %j] : memref<3x5xf64>
        %2 = arith.addf %0, %1 : f64
        affine.store %2, %res[%i, %j] : memref<3x5xf64>
      }
    }

    affine.for %i = 0 to 3 {
      affine.for %j = 0 to 5 {
        %0 = affine.load %res[%i, %j] : memref<3x5xf64>
        %1 = arith.mulf %0, %0 : f64
        affine.store %1, %res[%i, %j] : memref<3x5xf64>
      }
    }

    return %res : memref<3x5xf64>
  }
}
