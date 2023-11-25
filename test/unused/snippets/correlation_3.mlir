#map = affine_map<(d0) -> (d0 + 1)>

module {
  func.func @foo(%arg2: f64, %arg3: memref<5x3xf64>, %arg5: memref<3xf64>, %arg6: memref<3xf64>) -> memref<5x3xf64> attributes {llvm.emit_c_interface} {
    %arg3cpy = memref.alloc() : memref<5x3xf64>
    memref.copy %arg3, %arg3cpy : memref<5x3xf64> to memref<5x3xf64>

    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e-01 : f64

    affine.for %arg7 = 0 to 5 {
      affine.for %arg8 = 0 to 3 {
        %1 = affine.load %arg5[%arg8] : memref<3xf64>
        %2 = affine.load %arg3cpy[%arg7, %arg8] : memref<5x3xf64>
        %3 = arith.subf %2, %1 : f64
        affine.store %3, %arg3cpy[%arg7, %arg8] : memref<5x3xf64>
        %4 = affine.load %arg6[%arg8] : memref<3xf64>
        %5 = arith.mulf %arg2, %4 : f64
        %6 = arith.divf %3, %5 : f64
        affine.store %6, %arg3cpy[%arg7, %arg8] : memref<5x3xf64>
      }
    }
    return %arg3cpy : memref<5x3xf64>
  }
}
