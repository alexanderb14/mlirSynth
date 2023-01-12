func.func @foo(%arg4: f64, %arg5: f64, %arg7: memref<3x7xf64>, %arg8: memref<7x5xf64>, %arg9: memref<5x9xf64>, %arg10: memref<3x9xf64>) -> memref<3x9xf64> attributes {llvm.emit_c_interface} {
  %arg6 = memref.alloc() : memref<3x5xf64>
  %arg10cpy = memref.alloc() : memref<3x9xf64>
  memref.copy %arg10, %arg10cpy : memref<3x9xf64> to memref<3x9xf64>

  %cst = arith.constant 0.000000e+00 : f64
  affine.for %arg11 = 0 to 3 {
    affine.for %arg12 = 0 to 5 {
      affine.store %cst, %arg6[%arg11, %arg12] : memref<3x5xf64>
      affine.for %arg13 = 0 to 7 {
        %0 = affine.load %arg7[%arg11, %arg13] : memref<3x7xf64>
        %1 = arith.mulf %arg4, %0 : f64
        %2 = affine.load %arg8[%arg13, %arg12] : memref<7x5xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = affine.load %arg6[%arg11, %arg12] : memref<3x5xf64>
        %5 = arith.addf %4, %3 : f64
        affine.store %5, %arg6[%arg11, %arg12] : memref<3x5xf64>
      }
    }
  }
  affine.for %arg11 = 0 to 3 {
    affine.for %arg12 = 0 to 9 {
      %0 = affine.load %arg10cpy[%arg11, %arg12] : memref<3x9xf64>
      %1 = arith.mulf %0, %arg5 : f64
      affine.store %1, %arg10cpy[%arg11, %arg12] : memref<3x9xf64>
      affine.for %arg13 = 0 to 5 {
        %2 = affine.load %arg6[%arg11, %arg13] : memref<3x5xf64>
        %3 = affine.load %arg9[%arg13, %arg12] : memref<5x9xf64>
        %4 = arith.mulf %2, %3 : f64
        %5 = affine.load %arg10cpy[%arg11, %arg12] : memref<3x9xf64>
        %6 = arith.addf %5, %4 : f64
        affine.store %6, %arg10cpy[%arg11, %arg12] : memref<3x9xf64>
      }
    }
  }
  return %arg10cpy : memref<3x9xf64>
}
