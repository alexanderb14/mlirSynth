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
    return %arg0 : tensor<5x5xf64>
  }
}

