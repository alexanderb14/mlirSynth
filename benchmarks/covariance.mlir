#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_covariance(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<5x3xf64>, %arg4: memref<3x3xf64>, %arg5: memref<3xf64>) attributes {changed_sizes = "3:3,5:5", llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    affine.for %arg6 = 0 to 3 {
      affine.store %cst_0, %arg5[%arg6] : memref<3xf64>
      affine.for %arg7 = 0 to 5 {
        %3 = affine.load %arg3[%arg7, %arg6] : memref<5x3xf64>
        %4 = affine.load %arg5[%arg6] : memref<3xf64>
        %5 = arith.addf %4, %3 : f64
        affine.store %5, %arg5[%arg6] : memref<3xf64>
      }
      %1 = affine.load %arg5[%arg6] : memref<3xf64>
      %2 = arith.divf %1, %arg2 : f64
      affine.store %2, %arg5[%arg6] : memref<3xf64>
    }
    affine.for %arg6 = 0 to 5 {
      affine.for %arg7 = 0 to 3 {
        %1 = affine.load %arg5[%arg7] : memref<3xf64>
        %2 = affine.load %arg3[%arg6, %arg7] : memref<5x3xf64>
        %3 = arith.subf %2, %1 : f64
        affine.store %3, %arg3[%arg6, %arg7] : memref<5x3xf64>
      }
    }
    %0 = arith.subf %arg2, %cst : f64
    affine.for %arg6 = 0 to 3 {
      affine.for %arg7 = #map(%arg6) to 3 {
        affine.store %cst_0, %arg4[%arg6, %arg7] : memref<3x3xf64>
        affine.for %arg8 = 0 to 5 {
          %3 = affine.load %arg3[%arg8, %arg6] : memref<5x3xf64>
          %4 = affine.load %arg3[%arg8, %arg7] : memref<5x3xf64>
          %5 = arith.mulf %3, %4 : f64
          %6 = affine.load %arg4[%arg6, %arg7] : memref<3x3xf64>
          %7 = arith.addf %6, %5 : f64
          affine.store %7, %arg4[%arg6, %arg7] : memref<3x3xf64>
        }
        %1 = affine.load %arg4[%arg6, %arg7] : memref<3x3xf64>
        %2 = arith.divf %1, %0 : f64
        affine.store %2, %arg4[%arg6, %arg7] : memref<3x3xf64>
        affine.store %2, %arg4[%arg7, %arg6] : memref<3x3xf64>
      }
    }
    return
  }
}
