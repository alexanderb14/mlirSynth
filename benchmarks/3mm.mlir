module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<800x900xf64>, %arg6: memref<800x1000xf64>, %arg7: memref<1000x900xf64>, %arg8: memref<900x1100xf64>, %arg9: memref<900x1200xf64>, %arg10: memref<1200x1100xf64>, %arg11: memref<800x1100xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg12 = 0 to 800 {
      affine.for %arg13 = 0 to 900 {
        affine.store %cst, %arg5[%arg12, %arg13] : memref<800x900xf64>
        affine.for %arg14 = 0 to 1000 {
          %0 = affine.load %arg6[%arg12, %arg14] : memref<800x1000xf64>
          %1 = affine.load %arg7[%arg14, %arg13] : memref<1000x900xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %arg5[%arg12, %arg13] : memref<800x900xf64>
          %4 = arith.addf %3, %2 : f64
          affine.store %4, %arg5[%arg12, %arg13] : memref<800x900xf64>
        }
      }
    }
    affine.for %arg12 = 0 to 900 {
      affine.for %arg13 = 0 to 1100 {
        affine.store %cst, %arg8[%arg12, %arg13] : memref<900x1100xf64>
        affine.for %arg14 = 0 to 1200 {
          %0 = affine.load %arg9[%arg12, %arg14] : memref<900x1200xf64>
          %1 = affine.load %arg10[%arg14, %arg13] : memref<1200x1100xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %arg8[%arg12, %arg13] : memref<900x1100xf64>
          %4 = arith.addf %3, %2 : f64
          affine.store %4, %arg8[%arg12, %arg13] : memref<900x1100xf64>
        }
      }
    }
    affine.for %arg12 = 0 to 800 {
      affine.for %arg13 = 0 to 1100 {
        affine.store %cst, %arg11[%arg12, %arg13] : memref<800x1100xf64>
        affine.for %arg14 = 0 to 900 {
          %0 = affine.load %arg5[%arg12, %arg14] : memref<800x900xf64>
          %1 = affine.load %arg8[%arg14, %arg13] : memref<900x1100xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %arg11[%arg12, %arg13] : memref<800x1100xf64>
          %4 = arith.addf %3, %2 : f64
          affine.store %4, %arg11[%arg12, %arg13] : memref<800x1100xf64>
        }
      }
    }
    return
  }
}

