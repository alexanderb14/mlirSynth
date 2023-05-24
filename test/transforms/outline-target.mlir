// RUN: opt %s --outline-target | FileCheck %s

// CHECK:       func.func @target_0(%arg0: tensor<3x7xf64>, %arg1: tensor<7x5xf64>, %arg2: tensor<5x13xf64>, %arg3: tensor<13x11xf64>) -> tensor<3x11xf64> attributes {irsynth.target} {
// CHECK-NEXT:    %0 = stablehlo.dot %arg0, %arg1
// CHECK-NEXT:    %1 = stablehlo.dot %arg2, %arg3
// CHECK-NEXT:    %2 = stablehlo.dot %0, %1
// CHECK-NEXT:    return %2
// CHECK-NEXT:  }
module {
  func.func @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<3x5xf64>, %arg6: memref<3x7xf64>, %arg7: memref<7x5xf64>, %arg8: memref<5x11xf64>, %arg9: memref<5x13xf64>, %arg10: memref<13x11xf64>, %arg11: memref<3x11xf64>) -> memref<3x11xf64> attributes {changed_sizes = "1000:7,900:5,800:3,1200:13,1100:11", llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg7 : memref<7x5xf64>
    %1 = bufferization.to_tensor %arg6 : memref<3x7xf64>
    %2 = stablehlo.dot %1, %0 : (tensor<3x7xf64>, tensor<7x5xf64>) -> tensor<3x5xf64>
    %3 = bufferization.to_tensor %arg9 : memref<5x13xf64>
    %4 = bufferization.to_tensor %arg10 : memref<13x11xf64>
    %5 = stablehlo.dot %3, %4 : (tensor<5x13xf64>, tensor<13x11xf64>) -> tensor<5x11xf64>
    %6 = stablehlo.dot %2, %5 : (tensor<3x5xf64>, tensor<5x11xf64>) -> tensor<3x11xf64>
    %7 = bufferization.to_memref %6 : memref<3x11xf64>
    return %7 : memref<3x11xf64>
  }
}
