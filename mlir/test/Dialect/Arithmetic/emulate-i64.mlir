// RUN: mlir-opt -arith-emulate-i64 %s | FileCheck %s

// CHECK-LABEL: func @addi_same_i32
// CHECK-SAME:    ([[ARG:%.+]]: i32) -> i32
// CHECK-NEXT:    [[X:%.+]] = arith.addi [[ARG]], [[ARG]] : i32
// CHECK-NEXT:    return [[X]] : i32
func.func @addi_same_i32(%a : i32) -> i32 {
    %x = arith.addi %a, %a : i32
    return %x : i32
}

// CHECK-LABEL: func @addi_same_vector_i32
// CHECK-SAME:    ([[ARG:%.+]]: vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:    [[X:%.+]] = arith.addi [[ARG]], [[ARG]] : vector<2xi32>
// CHECK-NEXT:    return [[X]] : vector<2xi32>
func.func @addi_same_vector_i32(%a : vector<2xi32>) -> vector<2xi32> {
    %x = arith.addi %a, %a : vector<2xi32>
    return %x : vector<2xi32>
}

// CHECK-LABEL: func @addi_scalar_a_b
// CHECK-SAME:    ([[ARG0:%.+]]: vector<2xi32>, [[ARG1:%.+]]: vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:    [[LOW0:%.+]]   = vector.extract [[ARG0]][0] : vector<2xi32>
// CHECK-NEXT:    [[HIGH0:%.+]]  = vector.extract [[ARG0]][1] : vector<2xi32>
// CHECK-NEXT:    [[LOW1:%.+]]   = vector.extract [[ARG1]][0] : vector<2xi32>
// CHECK-NEXT:    [[HIGH1:%.+]]  = vector.extract [[ARG1]][1] : vector<2xi32>
// CHECK-NEXT:    [[SUM_L:%.+]], [[CB:%.+]] = arith.addi_carry [[LOW0]], [[LOW1]] : i32, i1
// CHECK-NEXT:    [[CARRY:%.+]]  = arith.extui [[CB]] : i1 to i32
// CHECK-NEXT:    [[SUM_H0:%.+]] = arith.addi [[CARRY]], [[HIGH0]] : i32
// CHECK-NEXT:    [[SUM_H1:%.+]] = arith.addi [[SUM_H0]], [[HIGH1]] : i32
// CHECK:         [[INS0:%.+]]   = vector.insert [[SUM_L]], {{%.+}} [0] : i32 into vector<2xi32>
// CHECK-NEXT:    [[INS1:%.+]]   = vector.insert [[SUM_H1]], [[INS0]] [1] : i32 into vector<2xi32>
// CHECK-NEXT:    return [[INS1]] : vector<2xi32>
func.func @addi_scalar_a_b(%a : i64, %b : i64) -> i64 {
    %x = arith.addi %a, %b : i64
    return %x : i64
}

// CHECK-LABEL: func @addi_vector_a_b
// CHECK-SAME:    ([[ARG0:%.+]]: vector<2x4xi32>, [[ARG1:%.+]]: vector<2x4xi32>) -> vector<2x4xi32>
// CHECK-NEXT:    [[LOW0:%.+]]   = vector.extract [[ARG0]][0] : vector<2x4xi32>
// CHECK-NEXT:    [[HIGH0:%.+]]  = vector.extract [[ARG0]][1] : vector<2x4xi32>
// CHECK-NEXT:    [[LOW1:%.+]]   = vector.extract [[ARG1]][0] : vector<2x4xi32>
// CHECK-NEXT:    [[HIGH1:%.+]]  = vector.extract [[ARG1]][1] : vector<2x4xi32>
// CHECK-NEXT:    [[SUM_L:%.+]], [[CB:%.+]] = arith.addi_carry [[LOW0]], [[LOW1]] : vector<4xi32>, vector<4xi1>
// CHECK-NEXT:    [[CARRY:%.+]]  = arith.extui [[CB]] : vector<4xi1> to vector<4xi32>
// CHECK-NEXT:    [[SUM_H0:%.+]] = arith.addi [[CARRY]], [[HIGH0]] : vector<4xi32>
// CHECK-NEXT:    [[SUM_H1:%.+]] = arith.addi [[SUM_H0]], [[HIGH1]] : vector<4xi32>
// CHECK:         [[INS0:%.+]]   = vector.insert [[SUM_L]], {{%.+}} [0] : vector<4xi32> into vector<2x4xi32>
// CHECK-NEXT:    [[INS1:%.+]]   = vector.insert [[SUM_H1]], [[INS0]] [1] : vector<4xi32> into vector<2x4xi32>
// CHECK-NEXT:    return [[INS1]] : vector<2x4xi32>
func.func @addi_vector_a_b(%a : vector<4xi64>, %b : vector<4xi64>) -> vector<4xi64> {
    %x = arith.addi %a, %b : vector<4xi64>
    return %x : vector<4xi64>
}

// CHECK-LABEL: func @identity
// CHECK-SAME:     ([[ARG:%.+]]: vector<2x4xi32>) -> vector<2x4xi32>
// CHECK-NEXT:     return [[ARG]] : vector<2x4xi32>
func.func @identity(%x : vector<4xi64>) -> vector<4xi64> {
    return %x : vector<4xi64>
}

// CHECK-LABEL: func @call
// CHECK-SAME:     ([[ARG:%.+]]: vector<2x4xi32>) -> vector<2x4xi32>
// CHECK-NEXT:     [[RES:%.+]] = call @identity([[ARG]]) : (vector<2x4xi32>) -> vector<2x4xi32>
// CHECK-NEXT:     return [[RES]] : vector<2x4xi32>
func.func @call(%a : vector<4xi64>) -> vector<4xi64> {
    %res = func.call @identity(%a) : (vector<4xi64>) -> vector<4xi64>
    return %res : vector<4xi64>
}

func.func @constant() -> i64 {
    %c = arith.constant 0 : i64
    return %c : i64
}
