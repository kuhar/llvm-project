// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: test_bitcast0
func.func @test_bitcast0(%arg0 : i64) -> f64 {
  %0 = arith.bitcast %arg0 : i64 to f64
  return %0 : f64
}

// CHECK-LABEL: test_bitcast_tensor0
func.func @test_bitcast_tensor0(%arg0 : tensor<8x8xi64>) -> tensor<8x8xf64> {
  %0 = arith.bitcast %arg0 : tensor<8x8xi64> to tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// CHECK-LABEL: test_bitcast_vector0
func.func @test_bitcast_vector0(%arg0 : vector<8xi64>) -> vector<8xf64> {
  %0 = arith.bitcast %arg0 : vector<8xi64> to vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: test_bitcast_scalable_vector0
func.func @test_bitcast_scalable_vector0(%arg0 : vector<[8]xi64>) -> vector<[8]xf64> {
  %0 = arith.bitcast %arg0 : vector<[8]xi64> to vector<[8]xf64>
  return %0 : vector<[8]xf64>
}

// CHECK-LABEL: test_bitcast1
func.func @test_bitcast1(%arg0 : f32) -> i32 {
  %0 = arith.bitcast %arg0 : f32 to i32
  return %0 : i32
}

// CHECK-LABEL: test_bitcast_tensor1
func.func @test_bitcast_tensor1(%arg0 : tensor<8x8xf32>) -> tensor<8x8xi32> {
  %0 = arith.bitcast %arg0 : tensor<8x8xf32> to tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}

// CHECK-LABEL: test_bitcast_vector1
func.func @test_bitcast_vector1(%arg0 : vector<8xf32>) -> vector<8xi32> {
  %0 = arith.bitcast %arg0 : vector<8xf32> to vector<8xi32>
  return %0 : vector<8xi32>
}

// CHECK-LABEL: test_bitcast_scalable_vector1
func.func @test_bitcast_scalable_vector1(%arg0 : vector<[8]xf32>) -> vector<[8]xi32> {
  %0 = arith.bitcast %arg0 : vector<[8]xf32> to vector<[8]xi32>
  return %0 : vector<[8]xi32>
}

// CHECK-LABEL: test_bitcast_int_vector1
func.func @test_bitcast_int_vector1(%arg0 : i64) -> vector<2xi32> {
  %0 = arith.bitcast %arg0 : i64 to vector<2xi32>
  return %0 : vector<2xi32>
}
// CHECK-LABEL: test_bitcast_vector_int
func.func @test_bitcast_vector_int(%arg0 : vector<2x1xi32>) -> i64 {
  %0 = arith.bitcast %arg0 : vector<2x1xi32> to i64
  return %0 : i64
}

// CHECK-LABEL: test_bitcast_vector_float
func.func @test_bitcast_vector_float(%arg0 : vector<2x1xi32>) -> f64 {
  %0 = arith.bitcast %arg0 : vector<2x1xi32> to f64
  return %0 : f64
}