module {
  func.func @fn_0(%arg4: f64, %arg5: f64, %arg6_orig: memref<3x5xf64>, %arg7: memref<3x7xf64>, %arg8: memref<7x5xf64>, %arg9: memref<5x11xf64>, %arg10_orig: memref<3x11xf64>) -> memref<3x11xf64> attributes {irsynth.original} {
    %arg10 = memref.alloc() : memref<3x11xf64>
    memref.copy %arg10_orig, %arg10 : memref<3x11xf64> to memref<3x11xf64>

    %arg6 = memref.alloc() : memref<3x5xf64>
    memref.copy %arg6_orig, %arg6 : memref<3x5xf64> to memref<3x5xf64>

    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg11 = 0 to 3 {
      affine.for %arg12 = 0 to 5 {
        affine.store %cst, %arg6[%arg11, %arg12] : memref<3x5xf64>
        affine.for %arg13 = 0 to 7 {
          %0 = affine.load %arg7[%arg11, %arg13] : memref<3x7xf64>
          %1 = arith.mulf %arg4, %0 : f64
          %2 = affine.load %arg8[%arg13, %arg12] : memref<7x5xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %arg6[%arg11, %arg12] : memref<3x5xf64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %arg6[%arg11, %arg12] : memref<3x5xf64>
        }
      }
    }
    affine.for %arg11 = 0 to 3 {
      affine.for %arg12 = 0 to 11 {
        %0 = affine.load %arg10[%arg11, %arg12] : memref<3x11xf64>
        %1 = arith.mulf %0, %arg5 : f64
        affine.store %1, %arg10[%arg11, %arg12] : memref<3x11xf64>
        affine.for %arg13 = 0 to 5 {
          %2 = affine.load %arg6[%arg11, %arg13] : memref<3x5xf64>
          %3 = affine.load %arg9[%arg13, %arg12] : memref<5x11xf64>
          %4 = arith.mulf %2, %3 : f64
          %5 = affine.load %arg10[%arg11, %arg12] : memref<3x11xf64>
          %6 = arith.addf %5, %4 : f64
          affine.store %6, %arg10[%arg11, %arg12] : memref<3x11xf64>
        }
      }
    }
    return %arg10 : memref<3x11xf64>
  }

  func.func @fn_0_raised(%arg4: tensor<f64>, %arg5: tensor<f64>, %arg6: tensor<3x5xf64>, %arg7: tensor<3x7xf64>, %arg8: tensor<7x5xf64>, %arg9: tensor<5x11xf64>, %arg10: tensor<3x11xf64>) -> tensor<3x11xf64> attributes {irsynth.raised} {
    %0 = stablehlo.dot %arg7, %arg8 : (tensor<3x7xf64>, tensor<7x5xf64>) -> tensor<3x5xf64>
    %1 = chlo.broadcast_multiply %arg4, %0 : (tensor<f64>, tensor<3x5xf64>) -> tensor<3x5xf64>

    %2 = chlo.broadcast_multiply %arg5, %arg10 : (tensor<f64>, tensor<3x11xf64>) -> tensor<3x11xf64>
    %3 = "stablehlo.dot_general"(%1, %arg9) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<3x5xf64>, tensor<5x11xf64>) -> tensor<3x11xf64>
    %4 = chlo.broadcast_add %3, %arg10 : (tensor<3x11xf64>, tensor<3x11xf64>) -> tensor<3x11xf64>
    return %4 : tensor<3x11xf64>
  }
}
