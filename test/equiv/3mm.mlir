// RUN: test-executor %s --print-output-black-white | FileCheck %s

// CHECK: Testing with IO OK
// CHECK: Checking with CBMC OK
module {
  func.func @fn_0(%arg5_orig: memref<3x5xf64>, %arg6: memref<3x7xf64>, %arg7: memref<7x5xf64>, %arg8_orig: memref<5x11xf64>, %arg9: memref<5x13xf64>, %arg10: memref<13x11xf64>, %arg11_orig: memref<3x11xf64>) -> memref<3x11xf64> attributes {changed_sizes = "1000:7,900:5,800:3,1200:13,1100:11", irsynth.original} {

    %arg5 = memref.alloc() : memref<3x5xf64>
    memref.copy %arg5_orig, %arg5 : memref<3x5xf64> to memref<3x5xf64>

    %arg8 = memref.alloc() : memref<5x11xf64>
    memref.copy %arg8_orig, %arg8 : memref<5x11xf64> to memref<5x11xf64>

    %arg11 = memref.alloc() : memref<3x11xf64>
    memref.copy %arg11_orig, %arg11 : memref<3x11xf64> to memref<3x11xf64>

    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg12 = 0 to 3 {
      affine.for %arg13 = 0 to 5 {
        affine.store %cst, %arg5[%arg12, %arg13] : memref<3x5xf64>
        affine.for %arg14 = 0 to 7 {
          %0 = affine.load %arg6[%arg12, %arg14] : memref<3x7xf64>
          %1 = affine.load %arg7[%arg14, %arg13] : memref<7x5xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %arg5[%arg12, %arg13] : memref<3x5xf64>
          %4 = arith.addf %3, %2 : f64
          affine.store %4, %arg5[%arg12, %arg13] : memref<3x5xf64>
        }
      }
    }
    affine.for %arg12 = 0 to 5 {
      affine.for %arg13 = 0 to 11 {
        affine.store %cst, %arg8[%arg12, %arg13] : memref<5x11xf64>
        affine.for %arg14 = 0 to 13 {
          %0 = affine.load %arg9[%arg12, %arg14] : memref<5x13xf64>
          %1 = affine.load %arg10[%arg14, %arg13] : memref<13x11xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %arg8[%arg12, %arg13] : memref<5x11xf64>
          %4 = arith.addf %3, %2 : f64
          affine.store %4, %arg8[%arg12, %arg13] : memref<5x11xf64>
        }
      }
    }
    affine.for %arg12 = 0 to 3 {
      affine.for %arg13 = 0 to 11 {
        affine.store %cst, %arg11[%arg12, %arg13] : memref<3x11xf64>
        affine.for %arg14 = 0 to 5 {
          %0 = affine.load %arg5[%arg12, %arg14] : memref<3x5xf64>
          %1 = affine.load %arg8[%arg14, %arg13] : memref<5x11xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %arg11[%arg12, %arg13] : memref<3x11xf64>
          %4 = arith.addf %3, %2 : f64
          affine.store %4, %arg11[%arg12, %arg13] : memref<3x11xf64>
        }
      }
    }
    return %arg11 : memref<3x11xf64>
  }

  func.func @fn_0_raised(%arg5: tensor<3x5xf64>, %arg6: tensor<3x7xf64>, %arg7: tensor<7x5xf64>, %arg8: tensor<5x11xf64>, %arg9: tensor<5x13xf64>, %arg10: tensor<13x11xf64>, %arg11: tensor<3x11xf64>) -> tensor<3x11xf64> attributes {changed_sizes = "1000:7,900:5,800:3,1200:13,1100:11", irsynth.raised} {
    %0 = "mhlo.dot" (%arg6, %arg7) : (tensor<3x7xf64>, tensor<7x5xf64>) -> tensor<3x5xf64>
    %1 = "mhlo.dot" (%arg9, %arg10) : (tensor<5x13xf64>, tensor<13x11xf64>) -> tensor<5x11xf64>
    %2 = "mhlo.dot" (%0, %1) : (tensor<3x5xf64>, tensor<5x11xf64>) -> tensor<3x11xf64>
    return %2: tensor<3x11xf64>
  }
}
