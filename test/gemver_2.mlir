func.func @foo(%arg2: f64, %arg3: memref<3x3xf64>, %arg9: memref<3xf64>, %arg10: memref<3xf64>) -> memref<3xf64> attributes {llvm.emit_c_interface} {
  %arg9cpy = memref.alloc() : memref<3xf64>
  memref.copy %arg9, %arg9cpy : memref<3xf64> to memref<3xf64>

  affine.for %arg12 = 0 to 3 {
    affine.for %arg13 = 0 to 3 {
      %0 = affine.load %arg9cpy[%arg12] : memref<3xf64>
      %1 = affine.load %arg3[%arg13, %arg12] : memref<3x3xf64>
      %2 = arith.mulf %arg2, %1 : f64
      %3 = affine.load %arg10[%arg13] : memref<3xf64>
      %4 = arith.mulf %2, %3 : f64
      %5 = arith.addf %0, %4 : f64
      affine.store %5, %arg9cpy[%arg12] : memref<3xf64>
    }
  }
  return %arg9cpy : memref<3xf64>
}
