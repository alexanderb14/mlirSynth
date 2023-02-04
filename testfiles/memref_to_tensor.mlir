module {
  func.func @foo(%arg0: memref<5x3xf64>) {
    // Convert the memref to a tensor.
    %0 = bufferization.to_tensor %arg0 : memref<5x3xf64>
    return
  }
}
