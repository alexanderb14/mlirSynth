#!/bin/bash

set -e

if ! [ -f "$PWD/../SynthIR/README.md" ]; then
  echo "ERROR: Run this script from the root of the repository."
  exit 1
fi

# Pull MLIR-HLO and MLIR.
git clone https://github.com/tensorflow/mlir-hlo.git
pushd mlir-hlo
git checkout $(cat ../build_tools/mlir_hlo_version.txt)

git clone https://github.com/llvm/llvm-project.git
pushd llvm-project
git checkout $(cat ../build_tools/llvm_version.txt)
git am < ../../build_tools/llvm_patches/add-trait-verification-function.patch
popd

# Build MLIR.
mkdir llvm-build
cmake -GNinja \
  "-H$PWD/llvm-project/llvm" \
  "-B$PWD/llvm-build" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
  -DLLVM_INCLUDE_TOOLS=ON \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_BUILD_TOOLS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DLLVM_PARALLEL_LINK_JOBS=1 -DLLVM_PARALLEL_COMPILE_JOBS=16 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$PWD/llvm-build" --target all --target mlir-cpu-runner

# Build MLIR-HLO.
mkdir build
pushd build
cmake .. -GNinja \
 -DLLVM_ENABLE_LLD=ON \
 -DCMAKE_BUILD_TYPE=RelWithDebInfo \
 -DLLVM_ENABLE_ASSERTIONS=On \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 -DMLIR_DIR=${PWD}/../llvm-build/lib/cmake/mlir
cmake --build .
popd

popd
