//===- VectorAttributes.td - Vector Dialect ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the attributes used in the Vector dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_IR_VECTOR_ATTRIBUTES
#define MLIR_DIALECT_VECTOR_IR_VECTOR_ATTRIBUTES

include "mlir/Dialect/Vector/IR/Vector.td"
include "mlir/IR/EnumAttr.td"

def Vector_IteratorType : I32EnumAttr<"IteratorType", "Iterator type", [
  I32EnumAttrCase<"parallel", 0>,
  I32EnumAttrCase<"reduction", 1>
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::vector";
}

def Vector_IteratorTypeEnum
    : EnumAttr<Vector_Dialect, Vector_IteratorType, "iterator_type"> {
    let assemblyFormat = "`<` $value `>`";
}

def Vector_IteratorTypeArrayAttr
    : TypedArrayAttrBase<Vector_IteratorTypeEnum,
                         "Iterator type should be an enum.">;

def PrintPunctuation : I32EnumAttr<"PrintPunctuation",
                                  "Punctuation for separating vectors or vector elements", [
  I32EnumAttrCase<"NoPunctuation", 0, "no_punctuation">,
  I32EnumAttrCase<"NewLine", 1, "newline">,
  I32EnumAttrCase<"Comma", 2, "comma">,
  I32EnumAttrCase<"Open", 3, "open">,
  I32EnumAttrCase<"Close", 4, "close">
]> {
  let cppNamespace = "::mlir::vector";
  let genSpecializedAttr = 0;
}

def Vector_PrintPunctuation : EnumAttr<Vector_Dialect, PrintPunctuation, "punctuation"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // MLIR_DIALECT_VECTOR_IR_VECTOR_ATTRIBUTES
