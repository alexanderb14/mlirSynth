#map = affine_map<(d0) -> (d0 + 1)>

module {
  func.func @fn_0(%arg0orig: memref<5x5xf64> {irsynth.symmetric}, %arg1: memref<5x3xf64>, %arg2: f64, %arg3: f64) -> memref<5x5xf64> attributes {irsynth.original} {
    %arg0 = memref.alloc() : memref<5x5xf64>
    memref.copy %arg0orig, %arg0 : memref<5x5xf64> to memref<5x5xf64>

    affine.for %arg4 = 0 to 5 {
      affine.for %arg5 = 0 to #map(%arg4) {
        %0 = affine.load %arg0[%arg4, %arg5] : memref<5x5xf64>
        %1 = arith.mulf %0, %arg2 : f64
        affine.store %1, %arg0[%arg4, %arg5] : memref<5x5xf64>
      }
      affine.for %arg5 = 0 to 3 {
        affine.for %arg6 = 0 to #map(%arg4) {
          %0 = affine.load %arg1[%arg4, %arg5] : memref<5x3xf64>
          %1 = arith.mulf %arg3, %0 : f64
          %2 = affine.load %arg1[%arg6, %arg5] : memref<5x3xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %arg0[%arg4, %arg6] : memref<5x5xf64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %arg0[%arg4, %arg6] : memref<5x5xf64>
        }
      }
    }
    return %arg0 : memref<5x5xf64>
  }

  func.func @fn_0_raised(%arg0: tensor<5x5xf64> {irsynth.symmetric}, %arg1: tensor<5x3xf64>, %arg2: f64, %arg3: f64) -> tensor<5x5xf64> attributes {irsynth.raised} {

    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<5x3xf64>) -> tensor<3x5xf64>
    %1 = "mhlo.dot"(%arg1, %0) : (tensor<5x3xf64>, tensor<3x5xf64>) -> tensor<5x5xf64>

    %2 = tensor.from_elements %arg3 : tensor<1xf64>
    %3 = "chlo.broadcast_multiply"(%1, %2) : (tensor<5x5xf64>, tensor<1xf64>) -> tensor<5x5xf64>

    %4 = tensor.from_elements %arg2 : tensor<1xf64>
    %5 = "chlo.broadcast_multiply"(%arg0, %4) : (tensor<5x5xf64>, tensor<1xf64>) -> tensor<5x5xf64>

    %6 = "mhlo.add"(%5, %3) : (tensor<5x5xf64>, tensor<5x5xf64>) -> tensor<5x5xf64>

    return %6 : tensor<5x5xf64>
  }
}
