#map = affine_map<(d0) -> (d0 + 1)>

func.func @foo(%arg3: memref<5x3xf64>) -> memref<3x3xf64> attributes {llvm.emit_c_interface} {
  %arg4 = memref.alloc() : memref<3x3xf64>

  %cst = arith.constant 1.000000e+00 : f64
  %cst_0 = arith.constant 0.000000e+00 : f64
  %cst_1 = arith.constant 1.000000e-01 : f64

  affine.for %arg7 = 0 to 4 {
    affine.store %cst, %arg4[%arg7, %arg7] : memref<3x3xf64>
    affine.for %arg8 = #map(%arg7) to 3 {
      affine.store %cst_0, %arg4[%arg7, %arg8] : memref<3x3xf64>
      affine.for %arg9 = 0 to 5 {
        %2 = affine.load %arg3[%arg9, %arg7] : memref<5x3xf64>
        %3 = affine.load %arg3[%arg9, %arg8] : memref<5x3xf64>
        %4 = arith.mulf %2, %3 : f64
        %5 = affine.load %arg4[%arg7, %arg8] : memref<3x3xf64>
        %6 = arith.addf %5, %4 : f64
        affine.store %6, %arg4[%arg7, %arg8] : memref<3x3xf64>
      }
      %1 = affine.load %arg4[%arg7, %arg8] : memref<3x3xf64>
      affine.store %1, %arg4[%arg8, %arg7] : memref<3x3xf64>
    }
  }
  affine.store %cst, %arg4[4, 4] : memref<3x3xf64>
  return %arg4 : memref<3x3xf64>
}
