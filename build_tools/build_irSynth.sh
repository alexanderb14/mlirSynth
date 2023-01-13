#!/bin/bash

set -e

if ! [ -f "$PWD/../irSynth/README.md" ]; then
  echo "ERROR: Run this script from the root of the repository."
  exit 1
fi

# Autogenerate arg tuple construction source file.
python3 build_tools/gen_ArgTuples.py \
  --max_operands 3 \
  --max_attributes 3 \
  --max_regions 3 \
  --output irSynth/enumeration/ArgTuples.cc
clang-format -i irSynth/enumeration/ArgTuples.cc --style=file

# Build irSynth.
mkdir build
pushd build
cmake .. -GNinja \
  -DLLVM_ENABLE_LLD=ON \
  -DMLIR_DIR=${PWD}/../mlir-hlo/llvm-build/lib/cmake/mlir \
  -DMHLO_DIR=${PWD}/../mlir-hlo/build/cmake/modules/CMakeFiles \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build .
popd
