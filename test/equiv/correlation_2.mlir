// RUN: test-executor %s --print-output-black-white | FileCheck %s

// CHECK: Testing with IO OK
// CHECK: Checking with CBMC OK
module {
  func.func @fn_2(%arg0: memref<3xf64>, %arg1: memref<5x3xf64>, %arg2: memref<3xf64>) -> memref<3xf64> attributes {irsynth.original} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloc = memref.alloc() : memref<3xf64>
    memref.copy %arg0, %alloc : memref<3xf64> to memref<3xf64>
    affine.for %arg3 = 0 to 3 {
      affine.store %cst, %alloc[%arg3] : memref<3xf64>
      affine.for %arg4 = 0 to 5 {
        %0 = affine.load %arg1[%arg4, %arg3] : memref<5x3xf64>
        %1 = affine.load %arg2[%arg3] : memref<3xf64>
        %2 = arith.subf %0, %1 : f64
        %3 = arith.mulf %2, %2 : f64
        %4 = affine.load %alloc[%arg3] : memref<3xf64>
        %5 = arith.addf %4, %3 : f64
        affine.store %5, %alloc[%arg3] : memref<3xf64>
      }
    }
    return %alloc : memref<3xf64>
  }

  func.func @fn_2_raised(%arg0: tensor<3xf64>, %arg1: tensor<5x3xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> attributes {irsynth.raised} {
    %0 = "chlo.broadcast_subtract"(%arg1, %arg2) : (tensor<5x3xf64>, tensor<3xf64>) -> tensor<5x3xf64>
    %1 = "chlo.broadcast_multiply"(%0, %0) : (tensor<5x3xf64>, tensor<5x3xf64>) -> tensor<5x3xf64>

    %50 = mhlo.constant dense<0.000000e+00> : tensor<f64>
    %51 = mhlo.reduce(%1 init: %50) across dimensions = [0] : (tensor<5x3xf64>, tensor<f64>) -> tensor<3xf64>
     reducer(%arg90: tensor<f64>, %arg91: tensor<f64>)  {
      %33 = mhlo.add %arg90, %arg91 : tensor<f64>
      mhlo.return %33 : tensor<f64>
    }

    return %51: tensor<3xf64>
  }
}
