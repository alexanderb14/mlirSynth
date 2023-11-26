// RUN: test-executor %s --print-output-black-white | FileCheck %s

// CHECK: Testing with IO OK
// CHECK: Checking with CBMC OK
#map = affine_map<(d0) -> (d0)>
module {
  func.func @fn_0(%arg1: f64, %arg4: f64, %arg0: memref<3x5xf64>, %arg2: memref<3x3xf64> {irsynth.symmetric}, %arg3: memref<3x5xf64>) -> memref<3x5xf64> attributes {irsynth.original} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloc0 = memref.alloc() : memref<f64>
    %alloc = memref.alloc() : memref<f64>
    memref.copy %alloc0, %alloc : memref<f64> to memref<f64>
    %alloc_0 = memref.alloc() : memref<3x5xf64>
    memref.copy %arg3, %alloc_0 : memref<3x5xf64> to memref<3x5xf64>
    affine.for %arg5 = 0 to 3 {
      affine.for %arg6 = 0 to 5 {
        affine.store %cst, %alloc[] : memref<f64>
        affine.for %arg7 = 0 to #map(%arg5) {
          %10 = affine.load %arg0[%arg5, %arg6] : memref<3x5xf64>
          %11 = arith.mulf %arg1, %10 : f64
          %12 = affine.load %arg2[%arg5, %arg7] : memref<3x3xf64>
          %13 = arith.mulf %11, %12 : f64
          %14 = affine.load %alloc_0[%arg7, %arg6] : memref<3x5xf64>
          %15 = arith.addf %14, %13 : f64
          affine.store %15, %alloc_0[%arg7, %arg6] : memref<3x5xf64>
          %16 = affine.load %arg0[%arg7, %arg6] : memref<3x5xf64>
          %17 = affine.load %arg2[%arg5, %arg7] : memref<3x3xf64>
          %18 = arith.mulf %16, %17 : f64
          %19 = affine.load %alloc[] : memref<f64>
          %20 = arith.addf %19, %18 : f64
          affine.store %20, %alloc[] : memref<f64>
        }
        %0 = affine.load %alloc_0[%arg5, %arg6] : memref<3x5xf64>
        %1 = arith.mulf %arg4, %0 : f64
        %2 = affine.load %arg0[%arg5, %arg6] : memref<3x5xf64>
        %3 = arith.mulf %arg1, %2 : f64
        %4 = affine.load %arg2[%arg5, %arg5] : memref<3x3xf64>
        %5 = arith.mulf %3, %4 : f64
        %6 = arith.addf %1, %5 : f64
        %7 = affine.load %alloc[] : memref<f64>
        %8 = arith.mulf %arg1, %7 : f64
        %9 = arith.addf %6, %8 : f64
        affine.store %9, %alloc_0[%arg5, %arg6] : memref<3x5xf64>
      }
    }
    return %alloc_0 : memref<3x5xf64>
  }
  func.func @fn_0_raised(%arg3: tensor<f64>, %arg1: tensor<f64>, %arg4: tensor<3x5xf64>, %arg2: tensor<3x3xf64>, %arg0: tensor<3x5xf64>) -> tensor<3x5xf64> attributes {irsynth.raised} {
    %0 = chlo.broadcast_multiply %arg0, %arg1 : (tensor<3x5xf64>, tensor<f64>) -> tensor<3x5xf64>
    %1 = chlo.broadcast_multiply %arg3, %arg4 : (tensor<f64>, tensor<3x5xf64>) -> tensor<3x5xf64>
    %2 = "mhlo.dot"(%arg2, %1) : (tensor<3x3xf64>, tensor<3x5xf64>) -> tensor<3x5xf64>
    %3 = chlo.broadcast_add %0, %2 : (tensor<3x5xf64>, tensor<3x5xf64>) -> tensor<3x5xf64>
    return %3 : tensor<3x5xf64>
  }
}
