// RUN: test-executor %s --print-output-black-white | FileCheck %s

// CHECK: Testing with IO OK
// CHECK: Checking with CBMC OK
module {
  func.func @fn_0(%arg1: f64, %arg2: f64, %arg3: memref<3x3xf64>, %arg4: memref<3x3xf64>, %arg5_orig: memref<3xf64>, %arg6: memref<3xf64>, %arg7_orig: memref<3xf64>) -> memref<3xf64> attributes {changed_sizes = "1300:3", irsynth.original} {

    %arg5 = memref.alloc() : memref<3xf64>
    memref.copy %arg5_orig, %arg5 : memref<3xf64> to memref<3xf64>

    %arg7 = memref.alloc() : memref<3xf64>
    memref.copy %arg7_orig, %arg7 : memref<3xf64> to memref<3xf64>

    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg8 = 0 to 3 {
      affine.store %cst, %arg5[%arg8] : memref<3xf64>
      affine.store %cst, %arg7[%arg8] : memref<3xf64>
      affine.for %arg9 = 0 to 3 {
        %5 = affine.load %arg3[%arg8, %arg9] : memref<3x3xf64>
        %6 = affine.load %arg6[%arg9] : memref<3xf64>
        %7 = arith.mulf %5, %6 : f64
        %8 = affine.load %arg5[%arg8] : memref<3xf64>
        %9 = arith.addf %7, %8 : f64
        affine.store %9, %arg5[%arg8] : memref<3xf64>
        %10 = affine.load %arg4[%arg8, %arg9] : memref<3x3xf64>
        %11 = affine.load %arg6[%arg9] : memref<3xf64>
        %12 = arith.mulf %10, %11 : f64
        %13 = affine.load %arg7[%arg8] : memref<3xf64>
        %14 = arith.addf %12, %13 : f64
        affine.store %14, %arg7[%arg8] : memref<3xf64>
      }
      %0 = affine.load %arg5[%arg8] : memref<3xf64>
      %1 = arith.mulf %arg1, %0 : f64
      %2 = affine.load %arg7[%arg8] : memref<3xf64>
      %3 = arith.mulf %arg2, %2 : f64
      %4 = arith.addf %1, %3 : f64
      affine.store %4, %arg7[%arg8] : memref<3xf64>
    }
    return %arg7 : memref<3xf64>
  }

  func.func @fn_0_raised(%arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<3x3xf64>, %arg4: tensor<3x3xf64>, %arg5: tensor<3xf64>, %arg6: tensor<3xf64>, %arg7: tensor<3xf64>) -> tensor<3xf64> attributes {changed_sizes = "1300:3", irsynth.raised} {
    %0 = "mhlo.dot" (%arg3, %arg6) : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %1 = "chlo.broadcast_multiply" (%0, %arg1) : (tensor<3xf64>, tensor<f64>) -> tensor<3xf64>

    %2 = "mhlo.dot" (%arg4, %arg6) : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %3 = "chlo.broadcast_multiply" (%2, %arg2) : (tensor<3xf64>, tensor<f64>) -> tensor<3xf64>

    %4 = "mhlo.add" (%1, %3) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>

    return %4: tensor<3xf64>
  }
}
