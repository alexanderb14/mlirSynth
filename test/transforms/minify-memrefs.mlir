// RUN: opt %s --change-sizes | FileCheck %s

// CHECK:       func.func @foo(%arg0: memref<3x5xf64>, %arg1: memref<3x5xf64>) -> memref<3x5xf64> attributes {changed_sizes = "1000:3,2000:5"} {
// CHECK-NEXT:    %alloc = memref.alloc() : memref<3x5xf64>
// CHECK-NEXT:    affine.for %arg2 = 0 to 3 {
// CHECK-NEXT:      affine.for %arg3 = 0 to 5 {
// CHECK-NEXT:        %0 = affine.load %arg0[%arg2, %arg3] : memref<3x5xf64>
// CHECK-NEXT:        %1 = affine.load %arg1[%arg2, %arg3] : memref<3x5xf64>
// CHECK-NEXT:        %2 = arith.addf %0, %1 : f64
// CHECK-NEXT:        affine.store %2, %alloc[%arg2, %arg3] : memref<3x5xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return %alloc : memref<3x5xf64>
// CHECK-NEXT:  }
module {
  func.func @foo(%arg0: memref<1000x2000xf64>, %arg1: memref<1000x2000xf64>) -> memref<1000x2000xf64> {
    %res = memref.alloc() : memref<1000x2000xf64>

    affine.for %i = 0 to 1000 {
      affine.for %j = 0 to 2000 {
        %0 = affine.load %arg0[%i, %j] : memref<1000x2000xf64>
        %1 = affine.load %arg1[%i, %j] : memref<1000x2000xf64>
        %2 = arith.addf %0, %1 : f64
        affine.store %2, %res[%i, %j] : memref<1000x2000xf64>
      }
    }

    return %res : memref<1000x2000xf64>
  }
}
