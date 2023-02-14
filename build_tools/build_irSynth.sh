#!/bin/bash

set -e

if ! [ -f "$PWD/../irSynth/README.md" ]; then
  echo "ERROR: Run this script from the root of the repository."
  exit 1
fi

# Autogenerate arg tuple construction source file.
python3 build_tools/gen_CartesianProduct.py \
  --max_operands 3 \
  --max_attributes 2 \
  --max_regions 2 \
  --output irSynth/enumeration/CartesianProduct.cc
clang-format -i irSynth/enumeration/CartesianProduct.cc --style=file

# Configure irSynth build.
mkdir build
pushd build
cmake .. -GNinja \
  -DLLVM_ENABLE_LLD=ON \
  -DMLIR_DIR=${PWD}/../mlir-hlo/llvm-build/lib/cmake/mlir \
  -DMHLO_DIR=${PWD}/../mlir-hlo/build/cmake/modules/CMakeFiles \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
popd

# Generate OpInfos from tablegen.
pushd build
cmake --build . --target tablegen-opinfos
popd
TD_FILES="mlir-hlo/stablehlo/stablehlo/dialect/ChloOps.td \
mlir-hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td"
GEN_COMMON="-Imlir-hlo \
-Imlir-hlo/include \
-Imlir-hlo/include/mlir-hlo/Dialect/mhlo/IR \
-Imlir-hlo/stablehlo \
-Imlir-hlo/llvm-project/llvm/include \
-Imlir-hlo/llvm-project/mlir/include \
-Imlir-hlo/llvm-build/include \
-Imlir-hlo/llvm-build/tools/mlir/include"
cat $TD_FILES | ./build/bin/tablegen-opinfos $GEN_COMMON \
  -gen-op-info-decls -o irSynth/enumeration/OpInfos.h
cat $TD_FILES | ./build/bin/tablegen-opinfos $GEN_COMMON \
  -gen-op-info-defs -o irSynth/enumeration/OpInfos.cc

# Build irSynth.
pushd build
cmake --build .
popd

# Merge all compile_commands.json files, so that clangd can find them.
jq -s 'map(.[])' mlir-hlo/llvm-build/compile_commands.json \
  mlir-hlo/build/compile_commands.json \
  build/compile_commands.json \
  > compile_commands.json
