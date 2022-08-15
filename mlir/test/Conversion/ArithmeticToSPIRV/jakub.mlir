// RUN: mlir-opt -split-input-file -convert-arith-to-spirv -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// arithmetic ops
//===----------------------------------------------------------------------===//

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Int8, Int16, Int64, Float16, Float64, Shader], []>, #spv.resource_limits<>>
} {

// Check integer operation conversions.
// CHECK-LABEL: @int32_scalar
func.func @int32_scalar(%lhs: i32, %rhs: i32) {
  // CHECK: spv.IAdd %{{.*}}, %{{.*}}: i32
  %0 = arith.addi %lhs, %rhs: i32
  // CHECK: spv.ISub %{{.*}}, %{{.*}}: i32
  %1 = arith.subi %lhs, %rhs: i32
  // CHECK: spv.IMul %{{.*}}, %{{.*}}: i32
  %2 = arith.muli %lhs, %rhs: i32
  // CHECK: spv.SDiv %{{.*}}, %{{.*}}: i32
  %3 = arith.divsi %lhs, %rhs: i32
  // CHECK: spv.UDiv %{{.*}}, %{{.*}}: i32
  %4 = arith.divui %lhs, %rhs: i32
  // CHECK: spv.UMod %{{.*}}, %{{.*}}: i32
  %5 = arith.remui %lhs, %rhs: i32
  return
}
}
