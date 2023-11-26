#!/bin/bash

set -x
set -e

if ! [ -f "$PWD/../mlirSynth/README.md" ]; then
  echo "ERROR: Run this script from the root of the repository."
  exit 1
fi

# Preparations
# ####
# Parse build type
BUILD_DIR=build
BUILD_TYPE=RelWithDebInfo
if [ "$1" == "--debug" ]; then
  BUILD_DIR=build_debug
  BUILD_TYPE=Debug
fi

# Prepare deps directory
mkdir -p deps
pushd deps

# MLIR-HLO and MLIR
# ####
# Pull
if [ ! -d "mlir-hlo" ]; then
  git clone https://github.com/tensorflow/mlir-hlo.git
  pushd mlir-hlo
  git checkout abf4e4c1095fe17611437c3bed108dc60c9d92e0
  popd
fi

if [ ! -d "llvm-project" ]; then
  git clone https://github.com/llvm/llvm-project.git
  pushd llvm-project
  git checkout $(cat ../mlir-hlo/build_tools/llvm_version.txt)
  git am < ../../build_tools/patches/llvm/add-trait-verification-function.patch
  git am < ../../build_tools/patches/llvm/enable-emit-c-for-more-ops.patch
  popd
fi

# Build
pushd llvm-project
mkdir -p $BUILD_DIR
pushd $BUILD_DIR
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
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build .
popd
popd

# Build MLIR-HLO.
pushd mlir-hlo
mkdir -p $BUILD_DIR
pushd $BUILD_DIR
cmake .. \
  -GNinja \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DMLIR_DIR=${PWD}/../../llvm-project/${BUILD_DIR}/lib/cmake/mlir
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
if [ ! -d "indicators" ]; then
  git clone https://github.com/p-ranav/indicators
  pushd indicators
  git checkout ef71abd9bc7254f7734fa84d5b1c336be2deb9f7
  popd
fi

# Build
pushd indicators
python3 utils/amalgamate/amalgamate.py -c single_include.json -s .
popd

# CBMC
# ####
# Pull
if [ ! -d "cbmc" ]; then
  git clone https://github.com/diffblue/cbmc.git
  pushd cbmc
  git checkout b0dc2ea2f03c3a3817a03a91da9ccfd4b995794e
  git submodule update --init
  git am < ../../build_tools/patches/cbmc/use-fpa-theory-with-cvc5.patch
  popd
fi

# Build
pushd cbmc
mkdir -p build
pushd build
cmake .. \
  -GNinja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++
cmake --build .
popd

# CVC5
# ####
# Pull
if [ ! -d "cvc5" ]; then
  git clone https://github.com/cvc5/cvc5.git
  pushd cvc5
  git checkout 7d3c5a757b7da00045a6fc011cad94e70d8eb442
  ./configure.sh --auto-download --best
  popd

  # Build
  pushd cvc5
  mkdir -p build
  cd build
  make -j 16
  popd
fi

popd