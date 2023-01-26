// RUN: opt %s --copy-modified-memrefs | FileCheck %s

// CHECK:      foo

module {
  func.func @foo(%arg0: memref<3x5xf64>, %arg1: memref<3x5xf64>, %res: memref<3x5xf64>) -> memref<3x5xf64> attributes {irsynth.synthesize} {
    affine.for %i = 0 to 3 {
      affine.for %j = 0 to 5 {
        %0 = affine.load %arg0[%i, %j] : memref<3x5xf64>
        %1 = affine.load %arg1[%i, %j] : memref<3x5xf64>
        %2 = arith.addf %0, %1 : f64
        affine.store %2, %res[%i, %j] : memref<3x5xf64>
      }
    }

    return %res : memref<3x5xf64>
  }
}
