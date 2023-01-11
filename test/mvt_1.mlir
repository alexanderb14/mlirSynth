func.func @foo(%arg1: memref<3xf64>, %arg2: memref<3xf64>, %arg3: memref<3xf64>, %arg4: memref<3xf64>, %arg5: memref<3x3xf64>) -> memref<3xf64> attributes {llvm.emit_c_interface} {
  %arg1cpy = memref.alloc() : memref<3xf64>
  memref.copy %arg1, %arg1cpy : memref<3xf64> to memref<3xf64>

  affine.for %arg6 = 0 to 3 {
    affine.for %arg7 = 0 to 3 {
      %0 = affine.load %arg1[%arg6] : memref<3xf64>
      %1 = affine.load %arg5[%arg6, %arg7] : memref<3x3xf64>
      %2 = affine.load %arg3[%arg7] : memref<3xf64>
      %3 = arith.mulf %1, %2 : f64
      %4 = arith.addf %0, %3 : f64
      affine.store %4, %arg1[%arg6] : memref<3xf64>
    }
  }
  affine.for %arg6 = 0 to 3 {
    affine.for %arg7 = 0 to 3 {
      %0 = affine.load %arg2[%arg6] : memref<3xf64>
      %1 = affine.load %arg5[%arg7, %arg6] : memref<3x3xf64>
      %2 = affine.load %arg4[%arg7] : memref<3xf64>
      %3 = arith.mulf %1, %2 : f64
      %4 = arith.addf %0, %3 : f64
      affine.store %4, %arg2[%arg6] : memref<3xf64>
    }
  }
  return %arg1cpy : memref<3xf64>
}
