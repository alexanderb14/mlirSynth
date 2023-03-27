#!/bin/bash

cmake --build $(dirname $0)/../build

for f in $(find $(dirname $0) -name "*.mlir"); do
  echo "Running $f"
  $(dirname $0)/../build/bin/test-executor $f
done
