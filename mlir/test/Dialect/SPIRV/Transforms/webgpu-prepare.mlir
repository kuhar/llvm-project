// RUN: mlir-opt --split-input-file --verify-diagnostics \
// RUN:   --spirv-webgpu-prepare --cse %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.UMulExtended
//===----------------------------------------------------------------------===//

spirv.module Logical GLSL450 {

// CHECK-LABEL: func @umul_extended_i32
// CHECK-SAME:       ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32)
// CHECK-DAG:        [[CSTMASK:%.+]] = spirv.Constant 65535 : i32
// CHECK-DAG:        [[CST16:%.+]]   = spirv.Constant 16 : i32
// CHECK-NEXT:       [[RESLOW:%.+]]  = spirv.IMul [[ARG0]], [[ARG1]] : i32
// CHECK-NEXT:       [[LHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG0]], [[CSTMASK]] : i32
// CHECK-NEXT:       [[LHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG0]], [[CST16]] : i32
// CHECK-NEXT:       [[RHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG1]], [[CSTMASK]] : i32
// CHECK-NEXT:       [[RHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG1]], [[CST16]] : i32
// CHECK-DAG:        [[RESHI0:%.+]]  = spirv.IMul [[LHSHI]], [[RHSHI]] : i32
// CHECK-DAG:        [[MID0:%.+]]    = spirv.IMul [[LHSHI]], [[RHSLOW]] : i32
// CHECK-DAG:        [[MID1:%.+]]    = spirv.IMul [[LHSLOW]], [[RHSHI]] : i32
// CHECK-NEXT:       [[MID:%.+]]     = spirv.IAdd [[MID0]], [[MID1]] : i32
// CHECK-NEXT:       [[RESHI1:%.+]]  = spirv.ShiftRightLogical [[MID]], [[CST16]] : i32
// CHECK-NEXT:       [[RESHI:%.+]]   = spirv.IAdd [[RESHI0]], [[RESHI1]] : i32
// CHECK-NEXT:       [[RES:%.+]]     = spirv.CompositeConstruct [[RESLOW]], [[RESHI]] : (i32, i32) -> !spirv.struct<(i32, i32)>
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : !spirv.struct<(i32, i32)>
spirv.func @umul_extended_i32(%arg0 : i32, %arg1 : i32) -> !spirv.struct<(i32, i32)> "None" {
  %0 = spirv.UMulExtended %arg0, %arg1 : !spirv.struct<(i32, i32)>
  spirv.ReturnValue %0 : !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: func @umul_extended_vector_i32
// CHECK-SAME:       ([[ARG0:%.+]]: vector<3xi32>, [[ARG1:%.+]]: vector<3xi32>)
// CHECK-DAG:        [[CSTMASK:%.+]] = spirv.Constant dense<65535> : vector<3xi32>
// CHECK-DAG:        [[CST16:%.+]]   = spirv.Constant dense<16> : vector<3xi32>
// CHECK-NEXT:       [[RESLOW:%.+]]  = spirv.IMul [[ARG0]], [[ARG1]] : vector<3xi32>
// CHECK-NEXT:       [[LHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG0]], [[CSTMASK]] : vector<3xi32>
// CHECK-NEXT:       [[LHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG0]], [[CST16]] : vector<3xi32>
// CHECK-NEXT:       [[RHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG1]], [[CSTMASK]] : vector<3xi32>
// CHECK-NEXT:       [[RHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG1]], [[CST16]] : vector<3xi32>
// CHECK-DAG:        [[RESHI0:%.+]]  = spirv.IMul [[LHSHI]], [[RHSHI]] : vector<3xi32>
// CHECK-DAG:        [[MID0:%.+]]    = spirv.IMul [[LHSHI]], [[RHSLOW]] : vector<3xi32>
// CHECK-DAG:        [[MID1:%.+]]    = spirv.IMul [[LHSLOW]], [[RHSHI]] : vector<3xi32>
// CHECK-NEXT:       [[MID:%.+]]     = spirv.IAdd [[MID0]], [[MID1]] : vector<3xi32>
// CHECK-NEXT:       [[RESHI1:%.+]]  = spirv.ShiftRightLogical [[MID]], [[CST16]] : vector<3xi32>
// CHECK-NEXT:       [[RESHI:%.+]]   = spirv.IAdd [[RESHI0]], [[RESHI1]] : vector<3xi32>
// CHECK-NEXT:       [[RES:%.+]]     = spirv.CompositeConstruct [[RESLOW]], [[RESHI]]
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
spirv.func @umul_extended_vector_i32(%arg0 : vector<3xi32>, %arg1 : vector<3xi32>)
  -> !spirv.struct<(vector<3xi32>, vector<3xi32>)> "None" {
  %0 = spirv.UMulExtended %arg0, %arg1 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
  spirv.ReturnValue %0 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
}

// CHECK-LABEL: func @umul_extended_i16
// CHECK-NEXT:       spirv.UMulExtended
// CHECK-NEXT:       spirv.ReturnValue
spirv.func @umul_extended_i16(%arg : i16) -> !spirv.struct<(i16, i16)> "None" {
  %0 = spirv.UMulExtended %arg, %arg : !spirv.struct<(i16, i16)>
  spirv.ReturnValue %0 : !spirv.struct<(i16, i16)>
}

//===----------------------------------------------------------------------===//
// spirv.SMulExtended
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @smul_extended_i32
// CHECK-SAME:       ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32)
// CHECK-DAG:        [[CSTMASK:%.+]]  = spirv.Constant 65535 : i32
// CHECK-DAG:        [[CST31:%.+]]    = spirv.Constant 31 : i32
// CHECK-DAG:        [[CST16:%.+]]    = spirv.Constant 16 : i32
// CHECK-DAG:        [[LHSLOW:%.+]]   = spirv.BitwiseAnd [[ARG0]], [[CSTMASK]] : i32
// CHECK-DAG:        [[LHSHI:%.+]]    = spirv.ShiftRightLogical [[ARG0]], [[CST16]] : i32
// CHECK-DAG:        [[RHSLOW:%.+]]   = spirv.BitwiseAnd [[ARG1]], [[CSTMASK]] : i32
// CHECK-DAG:        [[RHSHI:%.+]]    = spirv.ShiftRightLogical [[ARG1]], [[CST16]] : i32
// CHECK-DAG:        [[RESLOW:%.+]]   = spirv.IMul [[ARG0]], [[ARG1]] : i32
// CHECK-DAG:        [[RESHI0:%.+]]   = spirv.IMul [[LHSHI]], [[RHSHI]] : i32
// CHECK-DAG:        [[MID0:%.+]]     = spirv.IMul [[LHSHI]], [[RHSLOW]] : i32
// CHECK-DAG:        [[MID1:%.+]]     = spirv.IMul [[LHSLOW]], [[RHSHI]] : i32
// CHECK-NEXT:       [[MID:%.+]]      = spirv.IAdd [[MID0]], [[MID1]] : i32
// CHECK-NEXT:       [[RESHI1:%.+]]   = spirv.ShiftRightLogical [[MID]], [[CST16]] : i32
// CHECK-NEXT:       [[UHI:%.+]]      = spirv.IAdd [[RESHI0]], [[RESHI1]] : i32
// CHECK-DAG:        [[LHSSIGN:%.+]]  = spirv.ShiftRightArithmetic [[ARG0]], [[CST31]] : i32
// CHECK-DAG:        [[RHSSIGN:%.+]]  = spirv.ShiftRightArithmetic [[ARG1]], [[CST31]] : i32
// CHECK-DAG:        [[SIGNHI0:%.+]]  = spirv.IMul [[LHSSIGN]], [[RHSLOW]] : i32
// CHECK-DAG:        [[SIGNHI1:%.+]]  = spirv.IMul [[RHSSIGN]], [[LHSLOW]] : i32
// CHECK-DAG:        [[SIGNTMP0:%.+]] = spirv.IMul [[LHSSIGN]], [[RHSHI]] : i32
// CHECK-DAG:        [[SIGNTMP1:%.+]] = spirv.IMul [[RHSSIGN]], [[LHSHI]] : i32
// CHECK-NEXT:       [[SIGNTMP2:%.+]] = spirv.IAdd [[SIGNTMP0]], [[SIGNTMP1]] : i32
// CHECK-NEXT:       [[SIGNHI2:%.+]]  = spirv.ShiftRightLogical [[SIGNTMP2]], [[CST16]] : i32
// CHECK-DAG:        [[INSTMP0:%.+]]  = spirv.IAdd [[UHI]], [[SIGNHI0]] : i32
// CHECK-DAG:        [[INSTMP1:%.+]]  = spirv.IAdd [[SIGNHI1]], [[SIGNHI2]] : i32
// CHECK-NEXT:       [[RESHI:%.+]]    = spirv.IAdd [[INSTMP0]], [[INSTMP1]] : i32
// CHECK-NEXT:       [[RES:%.+]]      = spirv.CompositeConstruct [[RESLOW]], [[RESHI]]
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : !spirv.struct<(i32, i32)>
spirv.func @smul_extended_i32(%arg0 : i32, %arg1 : i32) -> !spirv.struct<(i32, i32)> "None" {
  %0 = spirv.SMulExtended %arg0, %arg1 : !spirv.struct<(i32, i32)>
  spirv.ReturnValue %0 : !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: func @smul_extended_vector_i32
// CHECK-SAME:       ([[ARG0:%.+]]: vector<3xi32>, [[ARG1:%.+]]: vector<3xi32>)
// CHECK-DAG:                          spirv.Constant dense<65535> : vector<3xi32>
// CHECK-DAG:                          spirv.Constant dense<31> : vector<3xi32>
// CHECK-DAG:                          spirv.Constant dense<16> : vector<3xi32>
// CHECK:            [[RESLOW:%.+]]  = spirv.IMul [[ARG0]], [[ARG1]] : vector<3xi32>
// CHECK:                              spirv.IMul
// CHECK:                              spirv.IAdd
// CHECK:            [[RES:%.+]]     = spirv.CompositeConstruct [[RESLOW]], {{%.+}}
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
spirv.func @smul_extended_vector_i32(%arg0 : vector<3xi32>, %arg1 : vector<3xi32>)
  -> !spirv.struct<(vector<3xi32>, vector<3xi32>)> "None" {
  %0 = spirv.SMulExtended %arg0, %arg1 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
  spirv.ReturnValue %0 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
}

// CHECK-LABEL: func @smul_extended_i16
// CHECK-NEXT:       spirv.SMulExtended
// CHECK-NEXT:       spirv.ReturnValue
spirv.func @smul_extended_i16(%arg : i16) -> !spirv.struct<(i16, i16)> "None" {
  %0 = spirv.SMulExtended %arg, %arg : !spirv.struct<(i16, i16)>
  spirv.ReturnValue %0 : !spirv.struct<(i16, i16)>
}

} // end module
