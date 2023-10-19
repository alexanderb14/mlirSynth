#!/bin/bash

set -e

if ! [ -f "$PWD/../irSynth/README.md" ]; then
  echo "ERROR: Run this script from the root of the repository."
  exit 1
fi

# MLIR-HLO and MLIR
# ####
# Pull
git clone https://github.com/tensorflow/mlir-hlo.git
pushd mlir-hlo
git checkout abf4e4c1095fe17611437c3bed108dc60c9d92e0

git clone https://github.com/llvm/llvm-project.git
pushd llvm-project
git checkout $(cat ../build_tools/llvm_version.txt)
git am < ../../build_tools/llvm_patches/add-trait-verification-function.patch
git am < ../../build_tools/llvm_patches/enable-emit-c-for-more-ops.patch
popd

# Build
mkdir llvm-build
cmake -GNinja \
  "-H$PWD/llvm-project/llvm" \
  "-B$PWD/llvm-build" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DLLVM_INCLUDE_TOOLS=ON \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_BUILD_TOOLS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$PWD/llvm-build"

# Build MLIR-HLO.
mkdir build
pushd build
cmake .. -GNinja \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DMLIR_DIR=${PWD}/../llvm-build/lib/cmake/mlir
cmake --build .
popd
popd

# ISL
# ####
# Pull
wget -P /tmp https://libisl.sourceforge.io/isl-0.25.tar.gz
tar -xf /tmp/isl-0.25.tar.gz
rm /tmp/isl-0.25.tar.gz

# Build
pushd isl-0.25
CC=clang CXX=clang++ ./configure
make -j$(nproc)
popd
