#!/bin/bash

set -x
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
  --output mlirSynth/synthesis/CartesianProduct.cc
clang-format -i mlirSynth/synthesis/CartesianProduct.cc --style=file

# Configure mlirSynth build.
mkdir -p build
pushd build
cmake .. -GNinja \
  -DLLVM_ENABLE_LLD=ON \
  -DMLIR_DIR=${PWD}/../deps/llvm-project/build/lib/cmake/mlir \
  -DMHLO_DIR=${PWD}/../deps/mlir-hlo/build/cmake/modules/CMakeFiles \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
popd

# Generate Grammar from tablegen files.
pushd build
cmake --build . --target grammar-extractor
popd
TD_OPS="deps/mlir-hlo/stablehlo/stablehlo/dialect/ChloOps.td \
deps/mlir-hlo/stablehlo/stablehlo/dialect/StablehloOps.td \
deps/llvm-project/mlir/include/mlir/Dialect/Linalg/IR/LinalgOps.td \
deps/llvm-project/mlir/include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td"
TD_INCLUDES="-Ideps/mlir-hlo \
-Ideps/mlir-hlo/include \
-Ideps/mlir-hlo/include/mlir-hlo/Dialect/mhlo/IR \
-Ideps/mlir-hlo/stablehlo \
-Ideps/llvm-project/llvm/include \
-Ideps/llvm-project/mlir/include \
-Ideps/llvm-project/build/include \
-Ideps/llvm-project/build/tools/mlir/include \
-Ideps/llvm-project/mlir/include/mlir/Dialect/Linalg/IR"
cat $TD_OPS | ./build/bin/grammar-extractor $TD_INCLUDES \
  -gen-grammar-decls -o mlirSynth/synthesis/Grammar.h
cat $TD_OPS | ./build/bin/grammar-extractor $TD_INCLUDES \
  -gen-grammar-defs -o mlirSynth/synthesis/Grammar.cc

# Build mlirSynth.
pushd build
cmake --build .
popd

# Merge all compile_commands.json files, so that clangd can find them.
jq -s 'map(.[])' deps/llvm-project/build/compile_commands.json \
  deps/mlir-hlo/build/compile_commands.json \
  build/compile_commands.json \
  > compile_commands.json
