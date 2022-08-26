#! /usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
readonly MLIR="$1"
readonly TMP="$(mktemp -d)"

echo "tmp dir: $TMP"

cp "$MLIR" "$TMP/wide.mlir"
cp "$MLIR" "$TMP/emulated.mlir"
sed -i 's/@op_wide/@op_emulated/g' "$TMP/emulated.mlir"

bin/mlir-opt "$SCRIPT_DIR/harness.mlir" \
    --convert-vector-to-llvm -convert-scf-to-cf -convert-cf-to-llvm \
    --convert-func-to-llvm --convert-arith-to-llvm \
  | bin/mlir-translate --mlir-to-llvmir -o "$TMP/harness.ll"

bin/mlir-opt "$TMP/wide.mlir" \
    --convert-func-to-llvm --convert-arith-to-llvm \
  | bin/mlir-translate --mlir-to-llvmir -o "$TMP/wide.ll"

bin/mlir-opt "$TMP/emulated.mlir" \
    --arith-emulate-i64 \
    --convert-vector-to-llvm --convert-func-to-llvm --convert-arith-to-llvm \
  | bin/mlir-translate --mlir-to-llvmir -o "$TMP/emulated.ll"

bin/llvm-link "$TMP/harness.ll" "$TMP/wide.ll" "$TMP/emulated.ll" -S -o "$TMP/full.ll"

echo "$TMP/full.ll"
# bin/opt "$TMP/full.ll" --strip-debug -S

bin/llc "$TMP/full.ll" -o "$TMP/full.s"

clang++ -std=c++20 "$SCRIPT_DIR/check-emulation.cpp" "$TMP/full.s" -o ./check-emulation
