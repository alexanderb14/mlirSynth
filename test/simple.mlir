module {
  func.func @foo(%arg6 : memref<3x4xf64>, %arg9: memref<4x6xf64>) -> memref<3x4xf64> attributes { llvm.emit_c_interface} {
    return %arg6 : memref<3x4xf64>
  }
}
