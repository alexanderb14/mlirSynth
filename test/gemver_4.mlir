module {
  func.func @foo(%arg1: f64, %arg3: memref<3x3xf64>, %arg8: memref<3xf64>, %arg9: memref<3xf64>) -> memref<3xf64> attributes {llvm.emit_c_interface} {
    %arg8cpy = memref.alloc() : memref<3xf64>
    memref.copy %arg8, %arg8cpy : memref<3xf64> to memref<3xf64>

    affine.for %arg12 = 0 to 3 {
      affine.for %arg13 = 0 to 3 {
        %0 = affine.load %arg8cpy[%arg12] : memref<3xf64>
        %1 = affine.load %arg3[%arg12, %arg13] : memref<3x3xf64>
        %2 = arith.mulf %arg1, %1 : f64
        %3 = affine.load %arg9[%arg13] : memref<3xf64>
        %4 = arith.mulf %2, %3 : f64
        %5 = arith.addf %0, %4 : f64
        affine.store %5, %arg8cpy[%arg12] : memref<3xf64>
      }
    }
    return %arg8cpy : memref<3xf64>
  }
}
