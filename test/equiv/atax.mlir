// RUN: test-executor %s --print-output-black-white | FileCheck %s

// CHECK: Testing with IO OK
// CHECK: Checking with CBMC OK
module {
  func.func @fn_0(%arg2: memref<3x5xf64>, %arg3: memref<5xf64>, %arg5: memref<3xf64>) -> memref<5xf64> attributes {irsynth.original} {
    %arg4 = memref.alloc() : memref<5xf64>

    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg6 = 0 to 5 {
      affine.store %cst, %arg4[%arg6] : memref<5xf64>
    }
    affine.for %arg6 = 0 to 3 {
      affine.store %cst, %arg5[%arg6] : memref<3xf64>
      affine.for %arg7 = 0 to 5 {
        %0 = affine.load %arg5[%arg6] : memref<3xf64>
        %1 = affine.load %arg2[%arg6, %arg7] : memref<3x5xf64>
        %2 = affine.load %arg3[%arg7] : memref<5xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = arith.addf %0, %3 : f64
        affine.store %4, %arg5[%arg6] : memref<3xf64>
      }
      affine.for %arg7 = 0 to 5 {
        %0 = affine.load %arg4[%arg7] : memref<5xf64>
        %1 = affine.load %arg2[%arg6, %arg7] : memref<3x5xf64>
        %2 = affine.load %arg5[%arg6] : memref<3xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = arith.addf %0, %3 : f64
        affine.store %4, %arg4[%arg7] : memref<5xf64>
      }
    }
    return %arg4 : memref<5xf64>
  }

  func.func @fn_0_raised(%arg2: tensor<3x5xf64>, %arg3: tensor<5xf64>, %arg5: tensor<3xf64>) -> tensor<5xf64> attributes {irsynth.raised} {

    %1 = "mhlo.dot"(%arg2, %arg3) : (tensor<3x5xf64>, tensor<5xf64>) -> tensor<3xf64>
    %2 = "mhlo.dot"(%1, %arg2) : (tensor<3xf64>, tensor<3x5xf64>) -> tensor<5xf64>

    return %2: tensor<5xf64>
  }
}
