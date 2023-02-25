module {
  func.func @fn_0(%arg0: memref<5x3xf64>, %arg1: f64) -> memref<3xf64> attributes {irsynth.original} {
    %cst = arith.constant 0.000000e+00 : f64
    %alloc_0 = memref.alloc() : memref<3xf64>
    affine.for %arg2 = 0 to 3 {
      affine.store %cst, %alloc_0[%arg2] : memref<3xf64>
      affine.for %arg3 = 0 to 5 {
        %2 = affine.load %arg0[%arg3, %arg2] : memref<5x3xf64>
        %3 = affine.load %alloc_0[%arg2] : memref<3xf64>
        %4 = arith.addf %3, %2 : f64
        affine.store %4, %alloc_0[%arg2] : memref<3xf64>
      }
      %0 = affine.load %alloc_0[%arg2] : memref<3xf64>
      %1 = arith.divf %0, %arg1 : f64
      affine.store %1, %alloc_0[%arg2] : memref<3xf64>
    }
    return %alloc_0 : memref<3xf64>
  }
  func.func @fn_0_raised_0(%arg0: tensor<5x3xf64>, %arg1: tensor<f64>) -> tensor<3xf64> attributes {irsynth.raised} {
    %0 = chlo.broadcast_divide %arg0, %arg1 : (tensor<5x3xf64>, tensor<f64>) -> tensor<5x3xf64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f64>
    %2 = mhlo.reduce(%0 init: %1) applies mhlo.add across dimensions = [0] : (tensor<5x3xf64>, tensor<f64>) -> tensor<3xf64>
    return %2 : tensor<3xf64>
  }
  func.func @fn_0_raised_1(%arg0: tensor<5x3xf64>, %arg1: tensor<f64>) -> tensor<3xf64> attributes {irsynth.raised} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [0] : (tensor<5x3xf64>, tensor<f64>) -> tensor<3xf64>
    %2 = chlo.broadcast_divide %1, %arg1 : (tensor<3xf64>, tensor<f64>) -> tensor<3xf64>
    return %2 : tensor<3xf64>
  }
}
