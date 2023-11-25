module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_gesummv(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<1300x1300xf64>, %arg4: memref<1300x1300xf64>, %arg5: memref<1300xf64>, %arg6: memref<1300xf64>, %arg7: memref<1300xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg8 = 0 to 1300 {
      affine.store %cst, %arg5[%arg8] : memref<1300xf64>
      affine.store %cst, %arg7[%arg8] : memref<1300xf64>
      affine.for %arg9 = 0 to 1300 {
        %5 = affine.load %arg3[%arg8, %arg9] : memref<1300x1300xf64>
        %6 = affine.load %arg6[%arg9] : memref<1300xf64>
        %7 = arith.mulf %5, %6 : f64
        %8 = affine.load %arg5[%arg8] : memref<1300xf64>
        %9 = arith.addf %7, %8 : f64
        affine.store %9, %arg5[%arg8] : memref<1300xf64>
        %10 = affine.load %arg4[%arg8, %arg9] : memref<1300x1300xf64>
        %11 = affine.load %arg6[%arg9] : memref<1300xf64>
        %12 = arith.mulf %10, %11 : f64
        %13 = affine.load %arg7[%arg8] : memref<1300xf64>
        %14 = arith.addf %12, %13 : f64
        affine.store %14, %arg7[%arg8] : memref<1300xf64>
      }
      %0 = affine.load %arg5[%arg8] : memref<1300xf64>
      %1 = arith.mulf %arg1, %0 : f64
      %2 = affine.load %arg7[%arg8] : memref<1300xf64>
      %3 = arith.mulf %arg2, %2 : f64
      %4 = arith.addf %1, %3 : f64
      affine.store %4, %arg7[%arg8] : memref<1300xf64>
    }
    return
  }
}

