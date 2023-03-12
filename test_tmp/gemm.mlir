module {
  func.func @fn_0(%arg3: f64, %arg4: f64, %arg5_orig: memref<3x5xf64>, %arg6: memref<3x7xf64>, %arg7: memref<7x5xf64>) -> memref<3x5xf64> attributes {changed_sizes = "1000:3,1200:7,1100:5", irsynth.original} {
    %arg5 = memref.alloc() : memref<3x5xf64>
    memref.copy %arg5_orig, %arg5 : memref<3x5xf64> to memref<3x5xf64>

    affine.for %arg8 = 0 to 3 {
      affine.for %arg9 = 0 to 5 {
        %0 = affine.load %arg5[%arg8, %arg9] : memref<3x5xf64>
        %1 = arith.mulf %0, %arg4 : f64
        affine.store %1, %arg5[%arg8, %arg9] : memref<3x5xf64>
      }
      affine.for %arg9 = 0 to 7 {
        affine.for %arg10 = 0 to 5 {
          %0 = affine.load %arg6[%arg8, %arg9] : memref<3x7xf64>
          %1 = arith.mulf %arg3, %0 : f64
          %2 = affine.load %arg7[%arg9, %arg10] : memref<7x5xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %arg5[%arg8, %arg10] : memref<3x5xf64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %arg5[%arg8, %arg10] : memref<3x5xf64>
        }
      }
    }
    return %arg5 : memref<3x5xf64>
  }

  func.func @fn_0_raised(%arg3: tensor<f64>, %arg4: tensor<f64>, %arg5: tensor<3x5xf64>, %arg6: tensor<3x7xf64>, %arg7: tensor<7x5xf64>) -> tensor<3x5xf64> attributes {changed_sizes = "1000:3,1200:7,1100:5", irsynth.raised} {
    %0 = "mhlo.dot" (%arg6, %arg7) : (tensor<3x7xf64>, tensor<7x5xf64>) -> tensor<3x5xf64>
    %1 = "chlo.broadcast_multiply" (%0, %arg3) : (tensor<3x5xf64>, tensor<f64>) -> tensor<3x5xf64>
    %2 = "chlo.broadcast_multiply" (%arg5, %arg4) : (tensor<3x5xf64>, tensor<f64>) -> tensor<3x5xf64>
    %3 = "mhlo.add" (%2, %1) : (tensor<3x5xf64>, tensor<3x5xf64>) -> tensor<3x5xf64>
    return %3 : tensor<3x5xf64>
  }
}
