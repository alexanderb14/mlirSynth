module {
  func.func @fn_0(%arg1: f64, %arg2: f64, %arg3_orig: memref<3x3xf64>, %arg4: memref<3xf64>, %arg5: memref<3xf64>, %arg6: memref<3xf64>, %arg7: memref<3xf64>, %arg8_orig: memref<3xf64>, %arg9_orig: memref<3xf64>, %arg10: memref<3xf64>, %arg11: memref<3xf64>) -> memref<3xf64> attributes {changed_sizes = "4000:3", irsynth.original} {
    %arg3 = memref.alloc() : memref<3x3xf64>
    memref.copy %arg3_orig, %arg3 : memref<3x3xf64> to memref<3x3xf64>

    %arg8 = memref.alloc() : memref<3xf64>
    memref.copy %arg8_orig, %arg8 : memref<3xf64> to memref<3xf64>

    %arg9 = memref.alloc() : memref<3xf64>
    memref.copy %arg9_orig, %arg9 : memref<3xf64> to memref<3xf64>

    affine.for %arg12 = 0 to 3 {
      affine.for %arg13 = 0 to 3 {
        %0 = affine.load %arg3[%arg12, %arg13] : memref<3x3xf64>
        %1 = affine.load %arg4[%arg12] : memref<3xf64>
        %2 = affine.load %arg5[%arg13] : memref<3xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = arith.addf %0, %3 : f64
        %5 = affine.load %arg6[%arg12] : memref<3xf64>
        %6 = affine.load %arg7[%arg13] : memref<3xf64>
        %7 = arith.mulf %5, %6 : f64
        %8 = arith.addf %4, %7 : f64
        affine.store %8, %arg3[%arg12, %arg13] : memref<3x3xf64>
      }
    }
    affine.for %arg12 = 0 to 3 {
      affine.for %arg13 = 0 to 3 {
        %0 = affine.load %arg9[%arg12] : memref<3xf64>
        %1 = affine.load %arg3[%arg13, %arg12] : memref<3x3xf64>
        %2 = arith.mulf %arg2, %1 : f64
        %3 = affine.load %arg10[%arg13] : memref<3xf64>
        %4 = arith.mulf %2, %3 : f64
        %5 = arith.addf %0, %4 : f64
        affine.store %5, %arg9[%arg12] : memref<3xf64>
      }
    }
    affine.for %arg12 = 0 to 3 {
      %0 = affine.load %arg9[%arg12] : memref<3xf64>
      %1 = affine.load %arg11[%arg12] : memref<3xf64>
      %2 = arith.addf %0, %1 : f64
      affine.store %2, %arg9[%arg12] : memref<3xf64>
    }
    affine.for %arg12 = 0 to 3 {
      affine.for %arg13 = 0 to 3 {
        %0 = affine.load %arg8[%arg12] : memref<3xf64>
        %1 = affine.load %arg3[%arg12, %arg13] : memref<3x3xf64>
        %2 = arith.mulf %arg1, %1 : f64
        %3 = affine.load %arg9[%arg13] : memref<3xf64>
        %4 = arith.mulf %2, %3 : f64
        %5 = arith.addf %0, %4 : f64
        affine.store %5, %arg8[%arg12] : memref<3xf64>
      }
    }
    return %arg8 : memref<3xf64>
  }

  func.func @fn_0_raised(%arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<3x3xf64>, %arg4: tensor<3xf64>, %arg5: tensor<3xf64>, %arg6: tensor<3xf64>, %arg7: tensor<3xf64>, %arg8: tensor<3xf64>, %arg9: tensor<3xf64>, %arg10: tensor<3xf64>, %arg11: tensor<3xf64>) -> tensor<3xf64> attributes {changed_sizes = "4000:3", irsynth.raised} {
    // Arg3
    %0 = "mhlo.reshape"(%arg4) {new_sizes = dense<[3, 1]> : tensor<2xi64>} : (tensor<3xf64>) -> tensor<3x1xf64>
    %1 = "mhlo.reshape"(%arg5) {new_sizes = dense<[1, 3]> : tensor<2xi64>} : (tensor<3xf64>) -> tensor<1x3xf64>
    %2 = "mhlo.dot" (%0, %1) : (tensor<3x1xf64>, tensor<1x3xf64>) -> tensor<3x3xf64>

    %3 = "mhlo.reshape"(%arg6) {new_sizes = dense<[3, 1]> : tensor<2xi64>} : (tensor<3xf64>) -> tensor<3x1xf64>
    %4 = "mhlo.reshape"(%arg7) {new_sizes = dense<[1, 3]> : tensor<2xi64>} : (tensor<3xf64>) -> tensor<1x3xf64>
    %5 = "mhlo.dot" (%3, %4) : (tensor<3x1xf64>, tensor<1x3xf64>) -> tensor<3x3xf64>

    %6 = "mhlo.add"(%arg3, %2) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %7 = "mhlo.add"(%6, %5) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>

    // Arg9
    %10 = "chlo.broadcast_multiply"(%7, %arg2) : (tensor<3x3xf64>, tensor<f64>) -> tensor<3x3xf64>
    %11 = "mhlo.dot" (%arg10, %10) : (tensor<3xf64>, tensor<3x3xf64>) -> tensor<3xf64>
    %12 = "mhlo.add"(%arg9, %11) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>

    %13 = "mhlo.add"(%12, %arg11) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    
    // Arg8
    %14 = "chlo.broadcast_multiply"(%7, %arg1) : (tensor<3x3xf64>, tensor<f64>) -> tensor<3x3xf64>
    %15 = "mhlo.dot" (%14, %13) : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %16 = "mhlo.add"(%arg8, %15) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>

    return %16: tensor<3xf64>
  }
}
