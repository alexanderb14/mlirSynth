module {
  func.func @foo(%arg9: memref<3xf64>, %arg11: memref<3xf64>) -> memref<3xf64> attributes {llvm.emit_c_interface} {
    %arg9cpy = memref.alloc() : memref<3xf64>
    memref.copy %arg9, %arg9cpy : memref<3xf64> to memref<3xf64>

    affine.for %arg12 = 0 to 3 {
      %0 = affine.load %arg9cpy[%arg12] : memref<3xf64>
      %1 = affine.load %arg11[%arg12] : memref<3xf64>
      %2 = arith.addf %0, %1 : f64
      affine.store %2, %arg9cpy[%arg12] : memref<3xf64>
    }
    return %arg9cpy : memref<3xf64>
  }
}
