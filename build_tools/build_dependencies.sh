#!/bin/bash

set -x
set -e

if ! [ -f "$PWD/../mlirSynth/README.md" ]; then
  echo "ERROR: Run this script from the root of the repository."
  exit 1
fi

# Prepare deps directory
# ####
mkdir -p deps
pushd deps

# MLIR-HLO and MLIR
# ####
# Pull
git clone https://github.com/tensorflow/mlir-hlo.git
pushd mlir-hlo
git checkout abf4e4c1095fe17611437c3bed108dc60c9d92e0
popd

git clone https://github.com/llvm/llvm-project.git
pushd llvm-project
git checkout $(cat ../mlir-hlo/build_tools/llvm_version.txt)
git am < ../../build_tools/llvm_patches/add-trait-verification-function.patch
git am < ../../build_tools/llvm_patches/enable-emit-c-for-more-ops.patch

# Build
mkdir -p build
pushd build
cmake ../llvm \
  -GNinja \
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
cmake --build .
popd
popd

# Build MLIR-HLO.
pushd mlir-hlo
mkdir -p build
pushd build
cmake .. \
  -GNinja \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DMLIR_DIR=${PWD}/../../llvm-project/build/lib/cmake/mlir
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

# Indicators
# ####
# Pull
git clone https://github.com/p-ranav/indicators
pushd indicators
git checkout ef71abd9bc7254f7734fa84d5b1c336be2deb9f7

# Build
python3 utils/amalgamate/amalgamate.py -c single_include.json -s .
popd

popd