//===- CombiningOpsUtils.td - Combining/Reduction op enums -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMBINING_OPS_UTILS
#define COMBINING_OPS_UTILS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

// The "kind" of combining function for contractions and reductions.
def COMBINING_KIND_ADD : I32BitEnumAttrCaseBit<"ADD", 0, "add">;
def COMBINING_KIND_MUL : I32BitEnumAttrCaseBit<"MUL", 1, "mul">;
def COMBINING_KIND_MINUI : I32BitEnumAttrCaseBit<"MINUI", 2, "minui">;
def COMBINING_KIND_MINSI : I32BitEnumAttrCaseBit<"MINSI", 3, "minsi">;
def COMBINING_KIND_MINF : I32BitEnumAttrCaseBit<"MINF", 4, "minf">;
def COMBINING_KIND_MAXUI : I32BitEnumAttrCaseBit<"MAXUI", 5, "maxui">;
def COMBINING_KIND_MAXSI : I32BitEnumAttrCaseBit<"MAXSI", 6, "maxsi">;
def COMBINING_KIND_MAXF : I32BitEnumAttrCaseBit<"MAXF", 7, "maxf">;
def COMBINING_KIND_AND : I32BitEnumAttrCaseBit<"AND", 8, "and">;
def COMBINING_KIND_OR  : I32BitEnumAttrCaseBit<"OR", 9, "or">;
def COMBINING_KIND_XOR : I32BitEnumAttrCaseBit<"XOR", 10, "xor">;
def COMBINING_KIND_MINIMUMF : I32BitEnumAttrCaseBit<"MINIMUMF", 11, "minimumf">;
def COMBINING_KIND_MAXIMUMF : I32BitEnumAttrCaseBit<"MAXIMUMF", 12, "maximumf">;

def CombiningKind : I32BitEnumAttr<
    "CombiningKind",
    "Kind of combining function for contractions and reductions",
    [COMBINING_KIND_ADD, COMBINING_KIND_MUL, COMBINING_KIND_MINUI,
     COMBINING_KIND_MINSI, COMBINING_KIND_MINF, COMBINING_KIND_MAXUI,
     COMBINING_KIND_MAXSI, COMBINING_KIND_MAXF, COMBINING_KIND_AND,
     COMBINING_KIND_OR, COMBINING_KIND_XOR,
     COMBINING_KIND_MAXIMUMF, COMBINING_KIND_MINIMUMF]> {
  let cppNamespace = "::mlir::utils";
  let genSpecializedAttr = 0;
}

#endif // COMBINING_OPS_UTILS
