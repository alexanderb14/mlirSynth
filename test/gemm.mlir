func.func @foo(%arg3: f64, %arg4: f64, %arg5: memref<3x5xf64>, %arg6: memref<3x7xf64>, %arg7: memref<7x5xf64>) -> memref<3x5xf64> attributes {llvm.emit_c_interface} {
  %arg5cpy = memref.alloc() : memref<3x5xf64>
  memref.copy %arg5, %arg5cpy : memref<3x5xf64> to memref<3x5xf64>

  affine.for %arg8 = 0 to 3 {
    affine.for %arg9 = 0 to 5 {
      %0 = affine.load %arg5cpy[%arg8, %arg9] : memref<3x5xf64>
      %1 = arith.mulf %0, %arg4 : f64
      affine.store %1, %arg5cpy[%arg8, %arg9] : memref<3x5xf64>
    }
    affine.for %arg9 = 0 to 7 {
      affine.for %arg10 = 0 to 5 {
        %0 = affine.load %arg6[%arg8, %arg9] : memref<3x7xf64>
        %1 = arith.mulf %arg3, %0 : f64
        %2 = affine.load %arg7[%arg9, %arg10] : memref<7x5xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = affine.load %arg5cpy[%arg8, %arg10] : memref<3x5xf64>
        %5 = arith.addf %4, %3 : f64
        affine.store %5, %arg5cpy[%arg8, %arg10] : memref<3x5xf64>
      }
    }
  }
  return %arg5cpy : memref<3x5xf64>
}
