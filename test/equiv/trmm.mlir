#map = affine_map<(d0) -> (d0 + 1)>

module {
  func.func @fn_0(%arg0: memref<3x3xf64> {irsynth.lower_triangular}, %arg1: memref<3x5xf64>) -> memref<3x5xf64> attributes {irsynth.original} {
    %alloc = memref.alloc() : memref<3x5xf64>
    memref.copy %arg1, %alloc : memref<3x5xf64> to memref<3x5xf64>
    affine.for %arg2 = 0 to 3 {
      affine.for %arg3 = 0 to 5 {
        affine.for %arg4 = #map(%arg2) to 3 {
          %0 = affine.load %arg0[%arg4, %arg2] : memref<3x3xf64>
          %1 = affine.load %alloc[%arg4, %arg3] : memref<3x5xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %alloc[%arg2, %arg3] : memref<3x5xf64>
          %4 = arith.addf %3, %2 : f64
          affine.store %4, %alloc[%arg2, %arg3] : memref<3x5xf64>
        }
      }
    }
    return %alloc : memref<3x5xf64>
  }

  func.func @fn_0_raised(%arg0: tensor<3x3xf64> {irsynth.lower_triangular}, %arg1: tensor<3x5xf64>) -> tensor<3x5xf64> attributes {irsynth.raised} {

    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %1 = "mhlo.dot"(%0, %arg1) : (tensor<3x3xf64>, tensor<3x5xf64>) -> tensor<3x5xf64>

    return %1 : tensor<3x5xf64>
  }
}
