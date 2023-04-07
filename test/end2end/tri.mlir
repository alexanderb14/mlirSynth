// RUN: synthesizer %s --num-threads=32 --ignore-equivalent-candidates --ops=chlo.broadcast_add,stablehlo.select | FileCheck %s

// CHECK:     stablehlo.select
#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @foo(%arg6 : memref<4x4xf64>, %arg9: memref<4x4xf64>) -> memref<4x4xf64> attributes { llvm.emit_c_interface} {
    %arg10 = memref.alloc() : memref<4x4xf64>
    memref.copy %arg6, %arg10 : memref<4x4xf64> to memref<4x4xf64>
    %cst = arith.constant 0.000000e+00 : f64

    affine.for %arg11 = 0 to 4 {
      affine.for %arg12 = 0 to #map(%arg11) {
         %2 = affine.load %arg6[%arg11, %arg12] : memref<4x4xf64>
         %3 = affine.load %arg9[%arg11, %arg12] : memref<4x4xf64>
         %4 = arith.addf %2, %3 : f64
         affine.store %4, %arg10[%arg11, %arg12] : memref<4x4xf64>
      }
    }
    return %arg10 : memref<4x4xf64>
  }
}
