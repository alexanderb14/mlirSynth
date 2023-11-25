// RUN: test-executor %s --print-output-black-white | FileCheck %s

// CHECK: Testing with IO OK
// CHECK: Checking with CBMC OK
module {
  func.func @fn_0(%arg2: memref<5x3xf64>, %arg3_orig: memref<3xf64>, %arg4_orig: memref<5xf64>, %arg5: memref<3xf64>, %arg6: memref<5xf64>) -> memref<3xf64> attributes {changed_sizes = "2100:5,1900:3", irsynth.original} {
    %arg3 = memref.alloc() : memref<3xf64>
    memref.copy %arg3_orig, %arg3 : memref<3xf64> to memref<3xf64>

    %arg4 = memref.alloc() : memref<5xf64>
    memref.copy %arg4_orig, %arg4 : memref<5xf64> to memref<5xf64>

    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg7 = 0 to 3 {
      affine.store %cst, %arg3[%arg7] : memref<3xf64>
    }
    affine.for %arg7 = 0 to 5 {
      affine.store %cst, %arg4[%arg7] : memref<5xf64>
      affine.for %arg8 = 0 to 3 {
        %0 = affine.load %arg3[%arg8] : memref<3xf64>
        %1 = affine.load %arg6[%arg7] : memref<5xf64>
        %2 = affine.load %arg2[%arg7, %arg8] : memref<5x3xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = arith.addf %0, %3 : f64
        affine.store %4, %arg3[%arg8] : memref<3xf64>
        %5 = affine.load %arg4[%arg7] : memref<5xf64>
        %6 = affine.load %arg2[%arg7, %arg8] : memref<5x3xf64>
        %7 = affine.load %arg5[%arg8] : memref<3xf64>
        %8 = arith.mulf %6, %7 : f64
        %9 = arith.addf %5, %8 : f64
        affine.store %9, %arg4[%arg7] : memref<5xf64>
      }
    }
    return %arg3 : memref<3xf64>
  }

  func.func @fn_0_raised(%arg2: tensor<5x3xf64>, %arg3: tensor<3xf64>, %arg4: tensor<5xf64>, %arg5: tensor<3xf64>, %arg6: tensor<5xf64>) -> tensor<3xf64> attributes {changed_sizes = "2100:5,1900:3", irsynth.raised} {
    %0 = "mhlo.dot" (%arg6, %arg2) : (tensor<5xf64>, tensor<5x3xf64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
