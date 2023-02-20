#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_syr2k(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<1200x1200xf64>, %arg5: memref<1200x1000xf64>, %arg6: memref<1200x1000xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg7 = 0 to 1200 {
      affine.for %arg8 = 0 to #map(%arg7) {
        %0 = affine.load %arg4[%arg7, %arg8] : memref<1200x1200xf64>
        %1 = arith.mulf %0, %arg3 : f64
        affine.store %1, %arg4[%arg7, %arg8] : memref<1200x1200xf64>
      }
    }
    affine.for %arg7 = 0 to 1200 {
      affine.for %arg8 = 0 to 1000 {
        affine.for %arg9 = 0 to #map(%arg7) {
          %0 = affine.load %arg5[%arg9, %arg8] : memref<1200x1000xf64>
          %1 = arith.mulf %0, %arg2 : f64
          %2 = affine.load %arg6[%arg7, %arg8] : memref<1200x1000xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %arg4[%arg7, %arg9] : memref<1200x1200xf64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %arg4[%arg7, %arg9] : memref<1200x1200xf64>
        }
      }
    }
    affine.for %arg7 = 0 to 1200 {
      affine.for %arg8 = 0 to 1000 {
        affine.for %arg9 = 0 to #map(%arg7) {
          %0 = affine.load %arg6[%arg9, %arg8] : memref<1200x1000xf64>
          %1 = arith.mulf %0, %arg2 : f64
          %2 = affine.load %arg5[%arg7, %arg8] : memref<1200x1000xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %arg4[%arg7, %arg9] : memref<1200x1200xf64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %arg4[%arg7, %arg9] : memref<1200x1200xf64>
        }
      }
    }
    return
  }
}

