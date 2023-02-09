#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @fn_0(%arg0: memref<5x5xf64> {irsynth.symmetric}, %arg1: memref<5x3xf64>, %arg2: f64, %arg3: f64) -> memref<5x5xf64> attributes {irsynth.original} {
    %alloc = memref.alloc() : memref<5x5xf64>
    memref.copy %arg0, %alloc : memref<5x5xf64> to memref<5x5xf64>
    affine.for %arg4 = 0 to 5 {
      affine.for %arg5 = 0 to #map(%arg4) {
        %0 = affine.load %alloc[%arg4, %arg5] : memref<5x5xf64>
        %1 = arith.mulf %0, %arg2 : f64
        affine.store %1, %alloc[%arg4, %arg5] : memref<5x5xf64>
      }
      affine.for %arg5 = 0 to 3 {
        affine.for %arg6 = 0 to #map(%arg4) {
          %0 = affine.load %arg1[%arg4, %arg5] : memref<5x3xf64>
          %1 = arith.mulf %arg3, %0 : f64
          %2 = affine.load %arg1[%arg6, %arg5] : memref<5x3xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %alloc[%arg4, %arg6] : memref<5x5xf64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %alloc[%arg4, %arg6] : memref<5x5xf64>
        }
      }
    }
    return %alloc : memref<5x5xf64>
  }
  func.func @fn_0_raised(%arg0: tensor<5x5xf64> {irsynth.symmetric}, %arg1: tensor<5x3xf64>, %arg2: f64, %arg3: f64) -> tensor<5x5xf64> attributes {irsynth.raised} {
    %from_elements = tensor.from_elements %arg3 : tensor<1xf64>
    %from_elements_0 = tensor.from_elements %arg2 : tensor<1xf64>
    
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<5x3xf64>) -> tensor<3x5xf64>
    %1 = "mhlo.dot"(%arg1, %0) : (tensor<5x3xf64>, tensor<3x5xf64>) -> tensor<5x5xf64>
    %2 = chlo.broadcast_multiply %1, %from_elements : (tensor<5x5xf64>, tensor<1xf64>) -> tensor<5x5xf64>
    %3 = chlo.broadcast_multiply %arg0, %from_elements_0 : (tensor<5x5xf64>, tensor<1xf64>) -> tensor<5x5xf64>
    %4 = mhlo.add %3, %2 : tensor<5x5xf64>

    %5 = mhlo.constant dense<[[true, false, false, false, false], [true, true, false, false, false], [true, true, true, false, false], [true, true, true, true, false], [true, true, true, true, true]]> : tensor<5x5xi1>
    %6 = mhlo.select %5, %4, %arg0 : tensor<5x5xi1>, tensor<5x5xf64>

    return %6 : tensor<5x5xf64>
  }
}

