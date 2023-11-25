module {
  func.func @foo(%arg6 : memref<3x4xf64>, %arg9: memref<4x6xf64>) -> memref<3x6xf64> attributes { llvm.emit_c_interface} {
    %arg10 = memref.alloc() : memref<3x6xf64>
    %cst = arith.constant 0.000000e+00 : f64

    affine.for %arg11 = 0 to 3 {
      affine.for %arg12 = 0 to 6 {
        affine.store %cst, %arg10[%arg11, %arg12] : memref<3x6xf64>
        affine.for %arg13 = 0 to 4 {
          %2 = affine.load %arg6[%arg11, %arg13] : memref<3x4xf64>
          %3 = affine.load %arg9[%arg13, %arg12] : memref<4x6xf64>
          %4 = arith.mulf %2, %3 : f64
          %5 = affine.load %arg10[%arg11, %arg12] : memref<3x6xf64>
          %6 = arith.addf %5, %4 : f64
          affine.store %6, %arg10[%arg11, %arg12] : memref<3x6xf64>
        }
      }
    }
    return %arg10 : memref<3x6xf64>
  }
}
