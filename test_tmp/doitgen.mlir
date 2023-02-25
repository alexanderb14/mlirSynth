module {
  func.func @fn_0(%arg0: memref<7xf64>, %arg1: memref<5x3x7xf64>, %arg2: memref<7x7xf64>) -> memref<5x3x7xf64> attributes {irsynth.original} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloc = memref.alloc() : memref<7xf64>
    memref.copy %arg0, %alloc : memref<7xf64> to memref<7xf64>
    %alloc_0 = memref.alloc() : memref<5x3x7xf64>
    memref.copy %arg1, %alloc_0 : memref<5x3x7xf64> to memref<5x3x7xf64>
    affine.for %arg3 = 0 to 5 {
      affine.for %arg4 = 0 to 3 {
        affine.for %arg5 = 0 to 7 {
          affine.store %cst, %alloc[%arg5] : memref<7xf64>
          affine.for %arg6 = 0 to 7 {
            %0 = affine.load %alloc_0[%arg3, %arg4, %arg6] : memref<5x3x7xf64>
            %1 = affine.load %arg2[%arg6, %arg5] : memref<7x7xf64>
            %2 = arith.mulf %0, %1 : f64
            %3 = affine.load %alloc[%arg5] : memref<7xf64>
            %4 = arith.addf %3, %2 : f64
            affine.store %4, %alloc[%arg5] : memref<7xf64>
          }
        }
        affine.for %arg5 = 0 to 7 {
          %0 = affine.load %alloc[%arg5] : memref<7xf64>
          affine.store %0, %alloc_0[%arg3, %arg4, %arg5] : memref<5x3x7xf64>
        }
      }
    }
    return %alloc_0 : memref<5x3x7xf64>
  }
  //func.func @fn_0_raised(%arg0: tensor<7xf64>, %arg1: tensor<5x3x7xf64>, %arg2: tensor<7x7xf64>) -> tensor<5x3x7xf64> attributes {irsynth.raised} {
  //  %0 = "mhlo.dot_general"(%arg1, %arg2) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [0]>} : (tensor<5x3x7xf64>, tensor<7x7xf64>) -> tensor<5x3x7xf64>
  //  return %0 : tensor<5x3x7xf64>
  //}

  func.func @fn_0_raised(%arg0: tensor<7xf64>, %arg1: tensor<5x3x7xf64>, %arg2: tensor<7x7xf64>) -> tensor<5x3x7xf64> attributes {irsynth.raised} {
    %0 = "mhlo.reshape"(%arg1) {new_sizes = dense<[5, 3, 1, 7]> : tensor<4xi64>} : (tensor<5x3x7xf64>) -> tensor<5x3x1x7xf64>
    %1 = "mhlo.dot_general"(%0, %arg2) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [0]>} : (tensor<5x3x1x7xf64>, tensor<7x7xf64>) -> tensor<5x3x1x7xf64>
    %2 = "mhlo.reshape"(%1) {new_sizes = dense<[5, 3, 7]> : tensor<3xi64>} : (tensor<5x3x1x7xf64>) -> tensor<5x3x7xf64>
    return %2 : tensor<5x3x7xf64>
  }
}
