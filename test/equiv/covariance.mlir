// RUN: test-executor %s --print-output-black-white | FileCheck %s

// CHECK: Testing with IO OK
// CHECK: Checking with CBMC OK
#map = affine_map<(d0) -> (d0)>

module {
  func.func @fn_0(%arg2: f64, %arg3_orig: memref<5x3xf64>) -> memref<3x3xf64> attributes {irsynth.original,changed_sizes = "1400:5,1200:3", llvm.linkage = #llvm.linkage<external>} {
    %arg3 = memref.alloc() : memref<5x3xf64>
    memref.copy %arg3_orig, %arg3 : memref<5x3xf64> to memref<5x3xf64>

    %arg4 = memref.alloc() : memref<3x3xf64>
    

    %arg5 = memref.alloc() : memref<3xf64>

    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    affine.for %arg6 = 0 to 3 {
      affine.store %cst_0, %arg5[%arg6] : memref<3xf64>
      affine.for %arg7 = 0 to 5 {
        %3 = affine.load %arg3[%arg7, %arg6] : memref<5x3xf64>
        %4 = affine.load %arg5[%arg6] : memref<3xf64>
        %5 = arith.addf %4, %3 : f64
        affine.store %5, %arg5[%arg6] : memref<3xf64>
      }
      %1 = affine.load %arg5[%arg6] : memref<3xf64>
      %2 = arith.divf %1, %arg2 : f64
      affine.store %2, %arg5[%arg6] : memref<3xf64>
    }
    affine.for %arg6 = 0 to 5 {
      affine.for %arg7 = 0 to 3 {
        %1 = affine.load %arg5[%arg7] : memref<3xf64>
        %2 = affine.load %arg3[%arg6, %arg7] : memref<5x3xf64>
        %3 = arith.subf %2, %1 : f64
        affine.store %3, %arg3[%arg6, %arg7] : memref<5x3xf64>
      }
    }
    %0 = arith.subf %arg2, %cst : f64
    affine.for %arg6 = 0 to 3 {
      affine.for %arg7 = #map(%arg6) to 3 {
        affine.store %cst_0, %arg4[%arg6, %arg7] : memref<3x3xf64>
        affine.for %arg8 = 0 to 5 {
          %3 = affine.load %arg3[%arg8, %arg6] : memref<5x3xf64>
          %4 = affine.load %arg3[%arg8, %arg7] : memref<5x3xf64>
          %5 = arith.mulf %3, %4 : f64
          %6 = affine.load %arg4[%arg6, %arg7] : memref<3x3xf64>
          %7 = arith.addf %6, %5 : f64
          affine.store %7, %arg4[%arg6, %arg7] : memref<3x3xf64>
        }
        %1 = affine.load %arg4[%arg6, %arg7] : memref<3x3xf64>
        %2 = arith.divf %1, %0 : f64
        affine.store %2, %arg4[%arg6, %arg7] : memref<3x3xf64>
        affine.store %2, %arg4[%arg7, %arg6] : memref<3x3xf64>
      }
    }
    return %arg4 : memref<3x3xf64>
  }

  func.func @fn_0_raised(%arg2: tensor<f64>, %arg3: tensor<5x3xf64>) -> tensor<3x3xf64> attributes {irsynth.raised,changed_sizes = "1400:5,1200:3", llvm.linkage = #llvm.linkage<external>} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = mhlo.reduce(%arg3 init: %0) across dimensions = [0] : (tensor<5x3xf64>, tensor<f64>) -> tensor<3xf64>
     reducer(%arg8: tensor<f64>, %arg9: tensor<f64>)  {
      %16 = mhlo.add %arg8, %arg9 : tensor<f64>
      mhlo.return %16 : tensor<f64>
    }
    %4 = "chlo.broadcast_divide"(%1, %arg2) : (tensor<3xf64>, tensor<f64>) -> tensor<3xf64>

    %7 = "chlo.broadcast_subtract"(%arg3, %4) : (tensor<5x3xf64>, tensor<3xf64>) -> tensor<5x3xf64>
    %8 = "mhlo.transpose"(%7) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<5x3xf64>) -> tensor<3x5xf64>
    %9 = "mhlo.dot"(%8, %7) : (tensor<3x5xf64>, tensor<5x3xf64>) -> tensor<3x3xf64>
    %11 = mhlo.constant dense<1.000000e+00> : tensor<f64>
    %12 = "chlo.broadcast_subtract"(%arg2, %11) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %15 = "chlo.broadcast_divide"(%9, %12) : (tensor<3x3xf64>, tensor<f64>) -> tensor<3x3xf64>

    return %15 : tensor<3x3xf64>
  }
}
