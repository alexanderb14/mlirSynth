// RUN: synthesizer %s --num-threads=32 --ignore-equivalent-candidates --ops=mhlo.dot,chlo.broadcast_add,chlo.broadcast_subtract | FileCheck %s

// CHECK:     "mhlo.dot"
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_atax(%arg0: i32, %arg1: i32, %arg2: memref<1900x2100xf64>, %arg3: memref<2100xf64>, %arg4: memref<2100xf64>, %arg5: memref<1900xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg6 = 0 to 2100 {
      affine.store %cst, %arg4[%arg6] : memref<2100xf64>
    }
    affine.for %arg6 = 0 to 1900 {
      affine.store %cst, %arg5[%arg6] : memref<1900xf64>
      affine.for %arg7 = 0 to 2100 {
        %0 = affine.load %arg5[%arg6] : memref<1900xf64>
        %1 = affine.load %arg2[%arg6, %arg7] : memref<1900x2100xf64>
        %2 = affine.load %arg3[%arg7] : memref<2100xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = arith.addf %0, %3 : f64
        affine.store %4, %arg5[%arg6] : memref<1900xf64>
      }
      affine.for %arg7 = 0 to 2100 {
        %0 = affine.load %arg4[%arg7] : memref<2100xf64>
        %1 = affine.load %arg2[%arg6, %arg7] : memref<1900x2100xf64>
        %2 = affine.load %arg5[%arg6] : memref<1900xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = arith.addf %0, %3 : f64
        affine.store %4, %arg4[%arg7] : memref<2100xf64>
      }
    }
    return
  }
}

