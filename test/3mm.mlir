func.func @foo(%arg6: memref<3x7xf64>, %arg7: memref<7x5xf64>, %arg9: memref<5x11xf64>, %arg10: memref<11x9xf64>) -> memref<3x9xf64> attributes { llvm.emit_c_interface } {
  %arg5 = memref.alloc() : memref<3x5xf64>
  %arg8 = memref.alloc() : memref<5x9xf64>
  %arg11 = memref.alloc() : memref<3x9xf64>

  %cst = arith.constant 0.000000e+00 : f64
  affine.for %arg12 = 0 to 3 {
    affine.for %arg13 = 0 to 5 {
      affine.store %cst, %arg5[%arg12, %arg13] : memref<3x5xf64>
      affine.for %arg14 = 0 to 7 {
        %0 = affine.load %arg6[%arg12, %arg14] : memref<3x7xf64>
        %1 = affine.load %arg7[%arg14, %arg13] : memref<7x5xf64>
        %2 = arith.mulf %0, %1 : f64
        %3 = affine.load %arg5[%arg12, %arg13] : memref<3x5xf64>
        %4 = arith.addf %3, %2 : f64
        affine.store %4, %arg5[%arg12, %arg13] : memref<3x5xf64>
      }
    }
  }
  affine.for %arg12 = 0 to 5 {
    affine.for %arg13 = 0 to 9 {
      affine.store %cst, %arg8[%arg12, %arg13] : memref<5x9xf64>
      affine.for %arg14 = 0 to 11 {
        %0 = affine.load %arg9[%arg12, %arg14] : memref<5x11xf64>
        %1 = affine.load %arg10[%arg14, %arg13] : memref<11x9xf64>
        %2 = arith.mulf %0, %1 : f64
        %3 = affine.load %arg8[%arg12, %arg13] : memref<5x9xf64>
        %4 = arith.addf %3, %2 : f64
        affine.store %4, %arg8[%arg12, %arg13] : memref<5x9xf64>
      }
    }
  }
  affine.for %arg12 = 0 to 3 {
    affine.for %arg13 = 0 to 9 {
      affine.store %cst, %arg11[%arg12, %arg13] : memref<3x9xf64>
      affine.for %arg14 = 0 to 5 {
        %0 = affine.load %arg5[%arg12, %arg14] : memref<3x5xf64>
        %1 = affine.load %arg8[%arg14, %arg13] : memref<5x9xf64>
        %2 = arith.mulf %0, %1 : f64
        %3 = affine.load %arg11[%arg12, %arg13] : memref<3x9xf64>
        %4 = arith.addf %3, %2 : f64
        affine.store %4, %arg11[%arg12, %arg13] : memref<3x9xf64>
      }
    }
  }
  return %arg11 : memref<3x9xf64>
}
