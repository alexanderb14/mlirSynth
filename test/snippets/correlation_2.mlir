#map = affine_map<(d0) -> (d0 + 1)>

module {
  func.func @foo(%arg2: f64, %arg3: memref<5x3xf64>, %arg5: memref<3xf64>) -> memref<3xf64> attributes {llvm.emit_c_interface} {
    %arg6 = memref.alloc() : memref<3xf64>

    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e-01 : f64
  
    affine.for %arg7 = 0 to 3 {
      affine.store %cst_0, %arg6[%arg7] : memref<3xf64>
      affine.for %arg8 = 0 to 5 {
        %6 = affine.load %arg3[%arg8, %arg7] : memref<5x3xf64>
        %7 = affine.load %arg5[%arg7] : memref<3xf64>
        %8 = arith.subf %6, %7 : f64
        %9 = arith.mulf %8, %8 : f64
        %10 = affine.load %arg6[%arg7] : memref<3xf64>
        %11 = arith.addf %10, %9 : f64
        affine.store %11, %arg6[%arg7] : memref<3xf64>
      }
      %1 = affine.load %arg6[%arg7] : memref<3xf64>
      %2 = arith.divf %1, %arg2 : f64
      %3 = math.sqrt %2 : f64
      %4 = arith.cmpf ole, %3, %cst_1 : f64
      %5 = arith.select %4, %cst, %3 : f64
      affine.store %5, %arg6[%arg7] : memref<3xf64>
    }
    return %arg6 : memref<3xf64>
  }
}
