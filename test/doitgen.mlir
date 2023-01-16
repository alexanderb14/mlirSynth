func.func @foo(%arg3: memref<5x3x7xf64>, %arg4: memref<7x7xf64>) -> memref<5x3x7xf64> attributes {llvm.emit_c_interface} {
  %arg3cpy = memref.alloc() : memref<5x3x7xf64>
  memref.copy %arg3, %arg3cpy : memref<5x3x7xf64> to memref<5x3x7xf64>

  %arg5 = memref.alloc() : memref<7xf64>

  %cst = arith.constant 0.000000e+00 : f64
  affine.for %arg6 = 0 to 5 {
    affine.for %arg7 = 0 to 3 {
      affine.for %arg8 = 0 to 7 {
        affine.store %cst, %arg5[%arg8] : memref<7xf64>
        affine.for %arg9 = 0 to 7 {
          %0 = affine.load %arg3cpy[%arg6, %arg7, %arg9] : memref<5x3x7xf64>
          %1 = affine.load %arg4[%arg9, %arg8] : memref<7x7xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %arg5[%arg8] : memref<7xf64>
          %4 = arith.addf %3, %2 : f64
          affine.store %4, %arg5[%arg8] : memref<7xf64>
        }
      }
      affine.for %arg8 = 0 to 7 {
        %0 = affine.load %arg5[%arg8] : memref<7xf64>
        affine.store %0, %arg3cpy[%arg6, %arg7, %arg8] : memref<5x3x7xf64>
      }
    }
  }
  return %arg3cpy : memref<5x3x7xf64>
}
