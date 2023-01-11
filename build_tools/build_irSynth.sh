#!/bin/bash

set -e

if ! [ -f "$PWD/../irSynth/README.md" ]; then
  echo "ERROR: Run this script from the root of the repository."
  exit 1
fi

# Build irSynth.
mkdir build
pushd build
cmake .. -GNinja \
  -DLLVM_ENABLE_LLD=ON \
  -DMLIR_DIR=${PWD}/../mlir-hlo/llvm-build/lib/cmake/mlir \
  -DMHLO_DIR=${PWD}/../mlir-hlo/build/cmake/modules/CMakeFiles \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build .
popd
