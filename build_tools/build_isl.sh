#!/bin/bash

set -e

if ! [ -f "$PWD/../irSynth/README.md" ]; then
  echo "ERROR: Run this script from the root of the repository."
  exit 1
fi

# Download and extract isl.
wget -P /tmp https://libisl.sourceforge.io/isl-0.25.tar.gz
tar -xf /tmp/isl-0.25.tar.gz
rm /tmp/isl-0.25.tar.gz

# Build isl.
pushd isl-0.25
CC=clang CXX=clang++ ./configure
make -j$(nproc)
popd
