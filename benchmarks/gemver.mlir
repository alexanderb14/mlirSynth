module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_gemver(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<2000x2000xf64>, %arg4: memref<2000xf64>, %arg5: memref<2000xf64>, %arg6: memref<2000xf64>, %arg7: memref<2000xf64>, %arg8: memref<2000xf64>, %arg9: memref<2000xf64>, %arg10: memref<2000xf64>, %arg11: memref<2000xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
//    affine.for %arg12 = 0 to 2000 {
//      affine.for %arg13 = 0 to 2000 {
//        %0 = affine.load %arg3[%arg12, %arg13] : memref<2000x2000xf64>
//        %1 = affine.load %arg4[%arg12] : memref<2000xf64>
//        %2 = affine.load %arg5[%arg13] : memref<2000xf64>
//        %3 = arith.mulf %1, %2 : f64
//        %4 = arith.addf %0, %3 : f64
//        %5 = affine.load %arg6[%arg12] : memref<2000xf64>
//        %6 = affine.load %arg7[%arg13] : memref<2000xf64>
//        %7 = arith.mulf %5, %6 : f64
//        %8 = arith.addf %4, %7 : f64
//        affine.store %8, %arg3[%arg12, %arg13] : memref<2000x2000xf64>
//      }
//    }
    affine.for %arg12 = 0 to 2000 {
      affine.for %arg13 = 0 to 2000 {
        %0 = affine.load %arg9[%arg12] : memref<2000xf64>
        %1 = affine.load %arg3[%arg13, %arg12] : memref<2000x2000xf64>
        %2 = arith.mulf %arg2, %1 : f64
        %3 = affine.load %arg10[%arg13] : memref<2000xf64>
        %4 = arith.mulf %2, %3 : f64
        %5 = arith.addf %0, %4 : f64
        affine.store %5, %arg9[%arg12] : memref<2000xf64>
      }
    }
    affine.for %arg12 = 0 to 2000 {
      %0 = affine.load %arg9[%arg12] : memref<2000xf64>
      %1 = affine.load %arg11[%arg12] : memref<2000xf64>
      %2 = arith.addf %0, %1 : f64
      affine.store %2, %arg9[%arg12] : memref<2000xf64>
    }
    affine.for %arg12 = 0 to 2000 {
      affine.for %arg13 = 0 to 2000 {
        %0 = affine.load %arg8[%arg12] : memref<2000xf64>
        %1 = affine.load %arg3[%arg12, %arg13] : memref<2000x2000xf64>
        %2 = arith.mulf %arg1, %1 : f64
        %3 = affine.load %arg9[%arg13] : memref<2000xf64>
        %4 = arith.mulf %2, %3 : f64
        %5 = arith.addf %0, %4 : f64
        affine.store %5, %arg8[%arg12] : memref<2000xf64>
      }
    }
    return
  }
}

