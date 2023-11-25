module {
  func.func @foo(%arg3: memref<5x3xf64>, %arg2: f64) -> memref<5x3xf64> attributes { llvm.emit_c_interface} {
    %arg5 = memref.alloc() : memref<5x3xf64>

    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e-01 : f64
    affine.for %arg7 = 0 to 3 {
      affine.for %arg8 = 0 to 5 {
        %3 = affine.load %arg3[%arg7, %arg8] : memref<5x3xf64>
        %4 = arith.addf %3, %3 : f64
        %5 = arith.subf %3, %4 : f64
        affine.store %5, %arg5[%arg7, %arg8] : memref<5x3xf64>
      }
    }
    return %arg5 : memref<5x3xf64>
  }
}
