#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_symm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<1000x1200xf64>, %arg5: memref<1000x1000xf64> {irsynth.symmetric}, %arg6: memref<1000x1200xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = memref.alloca() : memref<f64>
    %1 = llvm.mlir.undef : f64
    affine.store %1, %0[] : memref<f64>
    affine.for %arg7 = 0 to 1000 {
      affine.for %arg8 = 0 to 1200 {
        affine.store %cst, %0[] : memref<f64>
        affine.for %arg9 = 0 to #map(%arg7) {
          %12 = affine.load %arg6[%arg7, %arg8] : memref<1000x1200xf64>
          %13 = arith.mulf %arg2, %12 : f64
          %14 = affine.load %arg5[%arg7, %arg9] : memref<1000x1000xf64>
          %15 = arith.mulf %13, %14 : f64
          %16 = affine.load %arg4[%arg9, %arg8] : memref<1000x1200xf64>
          %17 = arith.addf %16, %15 : f64
          affine.store %17, %arg4[%arg9, %arg8] : memref<1000x1200xf64>
          %18 = affine.load %arg6[%arg9, %arg8] : memref<1000x1200xf64>
          %19 = affine.load %arg5[%arg7, %arg9] : memref<1000x1000xf64>
          %20 = arith.mulf %18, %19 : f64
          %21 = affine.load %0[] : memref<f64>
          %22 = arith.addf %21, %20 : f64
          affine.store %22, %0[] : memref<f64>
        }
        %2 = affine.load %arg4[%arg7, %arg8] : memref<1000x1200xf64>
        %3 = arith.mulf %arg3, %2 : f64
        %4 = affine.load %arg6[%arg7, %arg8] : memref<1000x1200xf64>
        %5 = arith.mulf %arg2, %4 : f64
        %6 = affine.load %arg5[%arg7, %arg7] : memref<1000x1000xf64>
        %7 = arith.mulf %5, %6 : f64
        %8 = arith.addf %3, %7 : f64
        %9 = affine.load %0[] : memref<f64>
        %10 = arith.mulf %arg2, %9 : f64
        %11 = arith.addf %8, %10 : f64
        affine.store %11, %arg4[%arg7, %arg8] : memref<1000x1200xf64>
      }
    }
    return
  }
}

