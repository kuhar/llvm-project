//===- VectorCombiningAttributes.td - Vector Dialect -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the combining attributes used in the Vector dialect.
// The based enums are defined in Dialect Utils, so we need to separate the
// attributes using them from the rest of the vector attributes to avoid
// generating enum definitions twice.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_IR_VECTOR_COMBININGATTRIBUTES
#define MLIR_DIALECT_VECTOR_IR_VECTOR_COMBININGATTRIBUTES

include "mlir/Dialect/Vector/IR/Vector.td"
include "mlir/Dialect/Utils/CombiningOpsUtils.td"
include "mlir/IR/EnumAttr.td"

/// An attribute that specifies the combining function for `vector.contract`,
/// and `vector.reduction`.
def Vector_CombiningKindAttr : EnumAttr<Vector_Dialect, CombiningKind, "kind"> {
  let assemblyFormat = "`<` $value `>`";
  let cppNamespace = "::mlir::vector";
}

#endif // MLIR_DIALECT_VECTOR_IR_VECTOR_COMBININGATTRIBUTES
