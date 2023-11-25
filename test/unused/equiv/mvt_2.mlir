module {
  func.func @fn_0(%arg1_orig: memref<3xf64>, %arg2_orig: memref<3xf64>, %arg3: memref<3xf64>, %arg4: memref<3xf64>, %arg5: memref<3x3xf64>) -> memref<3xf64> attributes {changed_sizes = "2000:3", irsynth.original} {
    %arg1 = memref.alloc() : memref<3xf64>
    memref.copy %arg1_orig, %arg1 : memref<3xf64> to memref<3xf64>

    %arg2 = memref.alloc() : memref<3xf64>
    memref.copy %arg2_orig, %arg2 : memref<3xf64> to memref<3xf64>

    affine.for %arg6 = 0 to 3 {
      affine.for %arg7 = 0 to 3 {
        %0 = affine.load %arg1[%arg6] : memref<3xf64>
        %1 = affine.load %arg5[%arg6, %arg7] : memref<3x3xf64>
        %2 = affine.load %arg3[%arg7] : memref<3xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = arith.addf %0, %3 : f64
        affine.store %4, %arg1[%arg6] : memref<3xf64>
      }
    }
    affine.for %arg6 = 0 to 3 {
      affine.for %arg7 = 0 to 3 {
        %0 = affine.load %arg2[%arg6] : memref<3xf64>
        %1 = affine.load %arg5[%arg7, %arg6] : memref<3x3xf64>
        %2 = affine.load %arg4[%arg7] : memref<3xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = arith.addf %0, %3 : f64
        affine.store %4, %arg2[%arg6] : memref<3xf64>
      }
    }
    return %arg2: memref<3xf64>
  }

  func.func @fn_0_raised(%arg1: tensor<3xf64>, %arg2: tensor<3xf64>, %arg3: tensor<3xf64>, %arg4: tensor<3xf64>, %arg5: tensor<3x3xf64>) -> tensor<3xf64> attributes {changed_sizes = "2000:3", irsynth.raised} {
    %0 = "mhlo.dot" (%arg4, %arg5) : (tensor<3xf64>, tensor<3x3xf64>) -> tensor<3xf64>
    %1 = "mhlo.add" (%arg2, %0) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    return %1 : tensor<3xf64>
  }
}
