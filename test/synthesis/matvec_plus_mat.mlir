// RUN: synthesizer %s --num-threads=32 --ignore-equivalent-candidates --ops=stablehlo.dot_general | FileCheck %s

// CHECK:     stablehlo.dot_general
module {
  func.func @kernel_gemver(%arg3: memref<2000x2000xf64>, %arg4: memref<2000xf64>, %arg5: memref<2000xf64>) {
    affine.for %arg12 = 0 to 2000 {
      affine.for %arg13 = 0 to 2000 {
        %0 = affine.load %arg3[%arg12, %arg13] : memref<2000x2000xf64>
        %1 = affine.load %arg4[%arg12] : memref<2000xf64>
        %2 = affine.load %arg5[%arg13] : memref<2000xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = arith.addf %0, %3 : f64
        affine.store %3, %arg3[%arg12, %arg13] : memref<2000x2000xf64>
      }
    }
    return
  }
}
