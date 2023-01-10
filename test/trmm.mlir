#map = affine_map<(d0) -> (d0 + 1)>
func.func @foo(%arg2: f64, %arg3: memref<3x3xf64>, %arg4: memref<3x5xf64>) -> memref<3x5xf64> attributes {llvm.emit_c_interface} {
  %arg4cpy = memref.alloc() : memref<3x5xf64>
  memref.copy %arg4, %arg4cpy : memref<3x5xf64> to memref<3x5xf64>

  affine.for %arg5 = 0 to 3 {
    affine.for %arg6 = 0 to 5 {
      affine.for %arg7 = #map(%arg5) to 3 {
        %2 = affine.load %arg3[%arg7, %arg5] : memref<3x3xf64>
        %3 = affine.load %arg4cpy[%arg7, %arg6] : memref<3x5xf64>
        %4 = arith.mulf %2, %3 : f64
        %5 = affine.load %arg4cpy[%arg5, %arg6] : memref<3x5xf64>
        %6 = arith.addf %5, %4 : f64
        affine.store %6, %arg4cpy[%arg5, %arg6] : memref<3x5xf64>
      }
      %0 = affine.load %arg4cpy[%arg5, %arg6] : memref<3x5xf64>
      %1 = arith.mulf %arg2, %0 : f64
      affine.store %1, %arg4cpy[%arg5, %arg6] : memref<3x5xf64>
    }
  }
  return %arg4cpy : memref<3x5xf64>
}
