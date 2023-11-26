// RUN: test-executor %s --print-output-black-white | FileCheck %s

// CHECK: Testing with IO OK
// CHECK: Checking with CBMC OK
module {
  func.func @fn_2(%arg0: memref<3xf64>, %arg1: memref<5x3xf64>, %arg2: memref<3xf64>) -> memref<5x3xf64> attributes {irsynth.original} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloc = memref.alloc() : memref<5x3xf64>
    affine.for %arg3 = 0 to 3 {
      affine.for %arg4 = 0 to 5 {
        affine.store %cst, %alloc[%arg4, %arg3] : memref<5x3xf64>
        %0 = affine.load %arg1[%arg4, %arg3] : memref<5x3xf64>
        %1 = affine.load %arg2[%arg3] : memref<3xf64>
        %2 = arith.subf %0, %1 : f64
        %3 = arith.mulf %2, %2 : f64
        affine.store %3, %alloc[%arg4, %arg3] : memref<5x3xf64>
      }
    }
    return %alloc : memref<5x3xf64>
  }

  func.func @fn_2_raised(%arg0: tensor<3xf64>, %arg1: tensor<5x3xf64>, %arg2: tensor<3xf64>) -> tensor<5x3xf64> attributes {irsynth.raised} {
    %0 = "chlo.broadcast_subtract"(%arg1, %arg2) : (tensor<5x3xf64>, tensor<3xf64>) -> tensor<5x3xf64>
    %1 = "chlo.broadcast_multiply"(%0, %0) : (tensor<5x3xf64>, tensor<5x3xf64>) -> tensor<5x3xf64>

    return %1 : tensor<5x3xf64>
  }
}
