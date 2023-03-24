#map = affine_map<(d0) -> (d0 + 1)>

module {
  func.func @fn_0(%arg2: f64, %arg3_orig: memref<5x3xf64>) -> memref<3x3xf64> attributes {irsynth.original,changed_sizes = "1400:5,1200:3"} {
    %arg3 = memref.alloc() : memref<5x3xf64>
    memref.copy %arg3_orig, %arg3 : memref<5x3xf64> to memref<5x3xf64>

    %arg4 = memref.alloc() : memref<3x3xf64>
    %arg5 = memref.alloc() : memref<3xf64>
    %arg6 = memref.alloc() : memref<3xf64>

    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e-01 : f64
    affine.for %arg7 = 0 to 3 {
      affine.store %cst_0, %arg5[%arg7] : memref<3xf64>
      affine.for %arg8 = 0 to 5 {
        %3 = affine.load %arg3[%arg8, %arg7] : memref<5x3xf64>
        %4 = affine.load %arg5[%arg7] : memref<3xf64>
        %5 = arith.addf %4, %3 : f64
        affine.store %5, %arg5[%arg7] : memref<3xf64>
      }
      %1 = affine.load %arg5[%arg7] : memref<3xf64>
      %2 = arith.divf %1, %arg2 : f64
      affine.store %2, %arg5[%arg7] : memref<3xf64>
    }
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
    %0 = math.sqrt %arg2 : f64
    affine.for %arg7 = 0 to 5 {
      affine.for %arg8 = 0 to 3 {
        %1 = affine.load %arg5[%arg8] : memref<3xf64>
        %2 = affine.load %arg3[%arg7, %arg8] : memref<5x3xf64>
        %3 = arith.subf %2, %1 : f64
        affine.store %3, %arg3[%arg7, %arg8] : memref<5x3xf64>
        %4 = affine.load %arg6[%arg8] : memref<3xf64>
        %5 = arith.mulf %0, %4 : f64
        %6 = arith.divf %3, %5 : f64
        affine.store %6, %arg3[%arg7, %arg8] : memref<5x3xf64>
      }
    }
    affine.for %arg7 = 0 to 3 {
      affine.store %cst, %arg4[%arg7, %arg7] : memref<3x3xf64>
      affine.for %arg8 = #map(%arg7) to 3 {
        affine.store %cst_0, %arg4[%arg7, %arg8] : memref<3x3xf64>
        affine.for %arg9 = 0 to 5 {
          %2 = affine.load %arg3[%arg9, %arg7] : memref<5x3xf64>
          %3 = affine.load %arg3[%arg9, %arg8] : memref<5x3xf64>
          %4 = arith.mulf %2, %3 : f64
          %5 = affine.load %arg4[%arg7, %arg8] : memref<3x3xf64>
          %6 = arith.addf %5, %4 : f64
          affine.store %6, %arg4[%arg7, %arg8] : memref<3x3xf64>
        }
        %1 = affine.load %arg4[%arg7, %arg8] : memref<3x3xf64>
        affine.store %1, %arg4[%arg8, %arg7] : memref<3x3xf64>
      }
    }

    return %arg4 : memref<3x3xf64>
  }

  func.func @fn_0_raised(%arg0: tensor<f64>, %arg1: tensor<5x3xf64>) -> tensor<3x3xf64> attributes {irsynth.raised,changed_sizes = "1400:5,1200:3"} {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<3x3xf64>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<3xf64>
    %2 = mhlo.constant dense<1.000000e-01> : tensor<3xf64>
    %4 = mhlo.constant dense<0.000000e+00> : tensor<f64>
    %5 = mhlo.reduce(%arg1 init: %4) across dimensions = [0] : (tensor<5x3xf64>, tensor<f64>) -> tensor<3xf64>
     reducer(%arg2: tensor<f64>, %arg3: tensor<f64>)  {
      %33 = mhlo.add %arg2, %arg3 : tensor<f64>
      mhlo.return %33 : tensor<f64>
    }
    %6 = "chlo.broadcast_divide"(%5, %arg0) : (tensor<3xf64>, tensor<f64>) -> tensor<3xf64>

    %7 = mhlo.reshape %6 : (tensor<3xf64>) -> tensor<1x3xf64>
    %8 = "mhlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x3xf64>) -> tensor<5x3xf64>
    %9 = mhlo.subtract %arg1, %8 : tensor<5x3xf64>
    %10 = mhlo.multiply %9, %9 : tensor<5x3xf64>
    %11 = mhlo.reduce(%10 init: %4) across dimensions = [0] : (tensor<5x3xf64>, tensor<f64>) -> tensor<3xf64>
     reducer(%arg2: tensor<f64>, %arg3: tensor<f64>)  {
      %33 = mhlo.add %arg2, %arg3 : tensor<f64>
      mhlo.return %33 : tensor<f64>
    }
    %12 = "chlo.broadcast_divide"(%11, %arg0) : (tensor<3xf64>, tensor<f64>) -> tensor<3xf64>
    %13 = mhlo.sqrt %12 : tensor<3xf64>
    %14 = mhlo.compare  LE, %13, %2,  FLOAT : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xi1>
    %15 = mhlo.select %14, %1, %13 : tensor<3xi1>, tensor<3xf64>
    %16 = mhlo.reshape %6 : (tensor<3xf64>) -> tensor<1x3xf64>
    %17 = "mhlo.broadcast_in_dim"(%16) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x3xf64>) -> tensor<5x3xf64>
    %18 = mhlo.subtract %arg1, %17 : tensor<5x3xf64>
    %19 = mhlo.sqrt %arg0 : tensor<f64>
    %20 = "mhlo.broadcast_in_dim"(%19) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f64>) -> tensor<3xf64>
    %21 = mhlo.multiply %20, %15 : tensor<3xf64>
    %22 = mhlo.reshape %21 : (tensor<3xf64>) -> tensor<1x3xf64>
    %23 = "mhlo.broadcast_in_dim"(%22) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x3xf64>) -> tensor<5x3xf64>
    %24 = mhlo.divide %18, %23 : tensor<5x3xf64>
    %25 = "mhlo.transpose"(%24) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<5x3xf64>) -> tensor<3x5xf64>
    %26 = "mhlo.dot"(%25, %24) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<3x5xf64>, tensor<5x3xf64>) -> tensor<3x3xf64>
    %27 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<3xi32>
    %28 = "mhlo.broadcast_in_dim"(%27) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<3x3xi32>
    %29 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<3xi32>
    %30 = "mhlo.broadcast_in_dim"(%29) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<3x3xi32>
    %31 = mhlo.compare  EQ, %28, %30,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
    %32 = mhlo.select %31, %0, %26 : tensor<3x3xi1>, tensor<3x3xf64>
    return %32 : tensor<3x3xf64>
  }
}
