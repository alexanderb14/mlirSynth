module {
  func.func @foo(%arg3: memref<3x3xf64>, %arg4: memref<3xf64>, %arg5: memref<3xf64>, %arg6: memref<3xf64>, %arg7: memref<3xf64>) -> memref<3x3xf64> attributes {llvm.emit_c_interface} {
    %arg3cpy = memref.alloc() : memref<3x3xf64>
    memref.copy %arg3, %arg3cpy : memref<3x3xf64> to memref<3x3xf64>

    affine.for %arg12 = 0 to 3 {
      affine.for %arg13 = 0 to 3 {
        %0 = affine.load %arg3cpy[%arg12, %arg13] : memref<3x3xf64>
        %1 = affine.load %arg4[%arg12] : memref<3xf64>
        %2 = affine.load %arg5[%arg13] : memref<3xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = arith.addf %0, %3 : f64
        %5 = affine.load %arg6[%arg12] : memref<3xf64>
        %6 = affine.load %arg7[%arg13] : memref<3xf64>
        %7 = arith.mulf %5, %6 : f64
        %8 = arith.addf %4, %7 : f64
        affine.store %8, %arg3cpy[%arg12, %arg13] : memref<3x3xf64>
      }
    }
    return %arg3cpy : memref<3x3xf64>
  }
}
