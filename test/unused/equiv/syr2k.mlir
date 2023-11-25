#map = affine_map<(d0) -> (d0 + 1)>

module {
  func.func @fn_0(%arg2: f64, %arg3: f64, %arg4_orig: memref<5x5xf64>, %arg5: memref<5x3xf64>, %arg6: memref<5x3xf64>) -> memref<5x5xf64> attributes {changed_sizes = "1000:3,1200:5", irsynth.original} {
    %arg4 = memref.alloc() : memref<5x5xf64>
    memref.copy %arg4_orig, %arg4: memref<5x5xf64> to memref<5x5xf64>

    affine.for %arg7 = 0 to 5 {
      affine.for %arg8 = 0 to #map(%arg7) {
        %0 = affine.load %arg4[%arg7, %arg8] : memref<5x5xf64>
        %1 = arith.mulf %0, %arg3 : f64
        affine.store %1, %arg4[%arg7, %arg8] : memref<5x5xf64>
      }
    }
    affine.for %arg7 = 0 to 5 {
      affine.for %arg8 = 0 to 3 {
        affine.for %arg9 = 0 to #map(%arg7) {
          %0 = affine.load %arg5[%arg9, %arg8] : memref<5x3xf64>
          %1 = arith.mulf %0, %arg2 : f64
          %2 = affine.load %arg6[%arg7, %arg8] : memref<5x3xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %arg4[%arg7, %arg9] : memref<5x5xf64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %arg4[%arg7, %arg9] : memref<5x5xf64>
        }
      }
    }
    affine.for %arg7 = 0 to 5 {
      affine.for %arg8 = 0 to 3 {
        affine.for %arg9 = 0 to #map(%arg7) {
          %0 = affine.load %arg6[%arg9, %arg8] : memref<5x3xf64>
          %1 = arith.mulf %0, %arg2 : f64
          %2 = affine.load %arg5[%arg7, %arg8] : memref<5x3xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %arg4[%arg7, %arg9] : memref<5x5xf64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %arg4[%arg7, %arg9] : memref<5x5xf64>
        }
      }
    }
    return %arg4 : memref<5x5xf64>
  }

  func.func @fn_0_raised_1(%arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<5x5xf64>, %arg5: tensor<5x3xf64>, %arg6: tensor<5x3xf64>) -> tensor<5x5xf64> attributes {changed_sizes = "1000:3,1200:5", irsynth.raised} {
    %0 = "chlo.broadcast_multiply"(%arg4, %arg3) : (tensor<5x5xf64>, tensor<f64>) -> tensor<5x5xf64>

    %1 = "chlo.broadcast_multiply"(%arg6, %arg2) : (tensor<5x3xf64>, tensor<f64>) -> tensor<5x3xf64>
    %2 = "mhlo.transpose"(%arg5) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<5x3xf64>) -> tensor<3x5xf64>
    %3 = "mhlo.dot"(%1, %2) : (tensor<5x3xf64>, tensor<3x5xf64>) -> tensor<5x5xf64>

    %4 = "chlo.broadcast_multiply"(%arg5, %arg2) : (tensor<5x3xf64>, tensor<f64>) -> tensor<5x3xf64>
    %5 = "mhlo.transpose"(%arg6) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<5x3xf64>) -> tensor<3x5xf64>
    %6 = "mhlo.dot"(%4, %5) : (tensor<5x3xf64>, tensor<3x5xf64>) -> tensor<5x5xf64>

    %18 = "mhlo.add"(%0, %3) : (tensor<5x5xf64>, tensor<5x5xf64>) -> tensor<5x5xf64>
    %19 = "mhlo.add"(%18, %6) : (tensor<5x5xf64>, tensor<5x5xf64>) -> tensor<5x5xf64>

    %21 = mhlo.constant dense<[[true, false, false, false, false], [true, true, false, false, false], [true, true, true, false, false], [true, true, true, true, false], [true, true, true, true, true]]> : tensor<5x5xi1>
    %22 = mhlo.select %21, %19, %arg4: tensor<5x5xi1>, tensor<5x5xf64>

    return %22: tensor<5x5xf64>
  }

  func.func @fn_0_raised_2(%arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<5x5xf64>, %arg5: tensor<5x3xf64>, %arg6: tensor<5x3xf64>) -> tensor<5x5xf64> attributes {changed_sizes = "1000:3,1200:5", irsynth.raised} {
    %0 = "chlo.broadcast_multiply"(%arg4, %arg3) : (tensor<5x5xf64>, tensor<f64>) -> tensor<5x5xf64>

    %1 = "chlo.broadcast_multiply"(%arg6, %arg2) : (tensor<5x3xf64>, tensor<f64>) -> tensor<5x3xf64>
    %2 = "mhlo.transpose"(%arg5) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<5x3xf64>) -> tensor<3x5xf64>
    %3 = "mhlo.dot"(%1, %2) : (tensor<5x3xf64>, tensor<3x5xf64>) -> tensor<5x5xf64>

    %4 = "chlo.broadcast_multiply"(%arg5, %arg2) : (tensor<5x3xf64>, tensor<f64>) -> tensor<5x3xf64>
    %5 = "mhlo.transpose"(%arg6) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<5x3xf64>) -> tensor<3x5xf64>
    %6 = "mhlo.dot"(%4, %5) : (tensor<5x3xf64>, tensor<3x5xf64>) -> tensor<5x5xf64>

    %18 = "mhlo.add"(%0, %3) : (tensor<5x5xf64>, tensor<5x5xf64>) -> tensor<5x5xf64>
    %19 = "mhlo.add"(%18, %6) : (tensor<5x5xf64>, tensor<5x5xf64>) -> tensor<5x5xf64>

    %25 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<5x5xi32>
    %26 = mhlo.constant dense<0> : tensor<i32>
    %27 = "mhlo.broadcast_in_dim"(%26) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i32>) -> tensor<5x5xi32>
    %28 = mhlo.add %25, %27 : tensor<5x5xi32>
    %29 = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<5x5xi32>
    %30 = mhlo.compare  GE, %28, %29,  SIGNED : (tensor<5x5xi32>, tensor<5x5xi32>) -> tensor<5x5xi1>

    %22 = mhlo.select %30, %19, %arg4: tensor<5x5xi1>, tensor<5x5xf64>

    return %22: tensor<5x5xf64>
  }
}
