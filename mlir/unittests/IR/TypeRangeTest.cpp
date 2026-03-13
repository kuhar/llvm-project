//===- TypeRangeTest.cpp - TypeRange/ValueRange unit tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeRange.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/Repeated.h"
#include "gtest/gtest.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// RepeatedType + TypeRange (low-level wrapper API)
//===----------------------------------------------------------------------===//

class RepeatedTypeRangeTest : public ::testing::Test {
protected:
  MLIRContext ctx;
  Type f32Ty = Float32Type::get(&ctx);
  Type i64Ty = IntegerType::get(&ctx, 64);
};

TEST_F(RepeatedTypeRangeTest, Basic) {
  RepeatedType rep{f32Ty};
  TypeRange range(rep, 4);

  EXPECT_EQ(range.size(), 4u);
  EXPECT_FALSE(range.empty());
  for (Type t : range)
    EXPECT_EQ(t, f32Ty);
}

TEST_F(RepeatedTypeRangeTest, IndexOperator) {
  RepeatedType rep{i64Ty};
  TypeRange range(rep, 5);

  for (size_t i = 0; i < 5; ++i)
    EXPECT_EQ(range[i], i64Ty);
}

TEST_F(RepeatedTypeRangeTest, Empty) {
  RepeatedType rep{f32Ty};
  TypeRange range(rep, 0);

  EXPECT_TRUE(range.empty());
  EXPECT_EQ(range.size(), 0u);
}

TEST_F(RepeatedTypeRangeTest, SingleElement) {
  RepeatedType rep{f32Ty};
  TypeRange range(rep, 1);

  EXPECT_EQ(range.size(), 1u);
  EXPECT_EQ(range.front(), f32Ty);
  EXPECT_EQ(range.back(), f32Ty);
}

TEST_F(RepeatedTypeRangeTest, Slice) {
  RepeatedType rep{f32Ty};
  TypeRange range(rep, 10);

  TypeRange sliced = range.slice(3, 4);
  EXPECT_EQ(sliced.size(), 4u);
  for (Type t : sliced)
    EXPECT_EQ(t, f32Ty);
}

TEST_F(RepeatedTypeRangeTest, DropFront) {
  RepeatedType rep{i64Ty};
  TypeRange range(rep, 5);

  TypeRange dropped = range.drop_front(2);
  EXPECT_EQ(dropped.size(), 3u);
  for (Type t : dropped)
    EXPECT_EQ(t, i64Ty);
}

TEST_F(RepeatedTypeRangeTest, DropBack) {
  RepeatedType rep{i64Ty};
  TypeRange range(rep, 5);

  TypeRange dropped = range.drop_back(3);
  EXPECT_EQ(dropped.size(), 2u);
  for (Type t : dropped)
    EXPECT_EQ(t, i64Ty);
}

TEST_F(RepeatedTypeRangeTest, TakeFront) {
  RepeatedType rep{f32Ty};
  TypeRange range(rep, 8);

  TypeRange taken = range.take_front(3);
  EXPECT_EQ(taken.size(), 3u);
  for (Type t : taken)
    EXPECT_EQ(t, f32Ty);
}

TEST_F(RepeatedTypeRangeTest, TakeBack) {
  RepeatedType rep{f32Ty};
  TypeRange range(rep, 8);

  TypeRange taken = range.take_back(2);
  EXPECT_EQ(taken.size(), 2u);
  for (Type t : taken)
    EXPECT_EQ(t, f32Ty);
}

TEST_F(RepeatedTypeRangeTest, EqualToMaterialized) {
  RepeatedType rep{f32Ty};
  TypeRange repeated(rep, 3);

  SmallVector<Type> materialized(3, f32Ty);
  TypeRange arrayBacked(materialized);

  EXPECT_TRUE(repeated == arrayBacked);
}

TEST_F(RepeatedTypeRangeTest, Hash) {
  RepeatedType rep{f32Ty};
  TypeRange repeated(rep, 3);

  SmallVector<Type> materialized(3, f32Ty);
  TypeRange arrayBacked(materialized);

  EXPECT_EQ(hash_value(repeated), hash_value(arrayBacked));
}

TEST_F(RepeatedTypeRangeTest, DenseMap) {
  RepeatedType rep{f32Ty};
  TypeRange repeated(rep, 2);

  SmallVector<Type> materialized(2, f32Ty);
  TypeRange arrayBacked(materialized);

  llvm::DenseMap<TypeRange, int> map;
  map[arrayBacked] = 42;
  EXPECT_EQ(map.count(repeated), 1u);
  EXPECT_EQ(map[repeated], 42);
}

//===----------------------------------------------------------------------===//
// llvm::Repeated<Type> implicit conversion to TypeRange
//===----------------------------------------------------------------------===//

TEST_F(RepeatedTypeRangeTest, RepeatedImplicitConversion) {
  llvm::Repeated<Type> rep(4, f32Ty);
  TypeRange range = rep;

  EXPECT_EQ(range.size(), 4u);
  for (Type t : range)
    EXPECT_EQ(t, f32Ty);
}

TEST_F(RepeatedTypeRangeTest, RepeatedAsArgument) {
  // Simulates passing Repeated<Type> to a function taking TypeRange.
  auto check = [&](TypeRange range) {
    EXPECT_EQ(range.size(), 3u);
    for (Type t : range)
      EXPECT_EQ(t, i64Ty);
  };
  check(llvm::Repeated<Type>(3, i64Ty));
}

TEST_F(RepeatedTypeRangeTest, RepeatedSlice) {
  llvm::Repeated<Type> rep(10, f32Ty);
  TypeRange range = rep;

  TypeRange sliced = range.slice(2, 5);
  EXPECT_EQ(sliced.size(), 5u);
  for (Type t : sliced)
    EXPECT_EQ(t, f32Ty);
}

TEST_F(RepeatedTypeRangeTest, RepeatedEqualToMaterialized) {
  llvm::Repeated<Type> rep(3, f32Ty);
  TypeRange repeated = rep;

  SmallVector<Type> materialized(3, f32Ty);
  TypeRange arrayBacked(materialized);

  EXPECT_TRUE(repeated == arrayBacked);
}

//===----------------------------------------------------------------------===//
// RepeatedValue + ValueRange (low-level wrapper API)
//===----------------------------------------------------------------------===//

namespace {

static Operation *createOp(MLIRContext *context,
                           ArrayRef<Type> resultTypes = {}) {
  context->allowUnregisteredDialects();
  return Operation::create(UnknownLoc::get(context),
                           OperationName("test.op", context), resultTypes, {},
                           NamedAttrList(), nullptr, {}, 0);
}

class RepeatedValueRangeTest : public ::testing::Test {
protected:
  MLIRContext ctx;
  Type f32Ty = Float32Type::get(&ctx);

  Value makeValue() {
    ops.push_back(createOp(&ctx, {f32Ty}));
    return ops.back()->getResult(0);
  }

  void TearDown() override {
    for (auto *op : llvm::reverse(ops))
      op->destroy();
  }

private:
  SmallVector<Operation *> ops;
};

TEST_F(RepeatedValueRangeTest, Basic) {
  Value v = makeValue();
  RepeatedValue rep{v};
  ValueRange range(rep, 4);

  EXPECT_EQ(range.size(), 4u);
  EXPECT_FALSE(range.empty());
  for (Value val : range)
    EXPECT_EQ(val, v);
}

TEST_F(RepeatedValueRangeTest, IndexOperator) {
  Value v = makeValue();
  RepeatedValue rep{v};
  ValueRange range(rep, 5);

  for (size_t i = 0; i < 5; ++i)
    EXPECT_EQ(range[i], v);
}

TEST_F(RepeatedValueRangeTest, Empty) {
  Value v = makeValue();
  RepeatedValue rep{v};
  ValueRange range(rep, 0);

  EXPECT_TRUE(range.empty());
  EXPECT_EQ(range.size(), 0u);
}

TEST_F(RepeatedValueRangeTest, Slice) {
  Value v = makeValue();
  RepeatedValue rep{v};
  ValueRange range(rep, 10);

  ValueRange sliced = range.slice(3, 4);
  EXPECT_EQ(sliced.size(), 4u);
  for (Value val : sliced)
    EXPECT_EQ(val, v);
}

TEST_F(RepeatedValueRangeTest, DropFront) {
  Value v = makeValue();
  RepeatedValue rep{v};
  ValueRange range(rep, 5);

  ValueRange dropped = range.drop_front(2);
  EXPECT_EQ(dropped.size(), 3u);
  for (Value val : dropped)
    EXPECT_EQ(val, v);
}

TEST_F(RepeatedValueRangeTest, DropBack) {
  Value v = makeValue();
  RepeatedValue rep{v};
  ValueRange range(rep, 5);

  ValueRange dropped = range.drop_back(3);
  EXPECT_EQ(dropped.size(), 2u);
  for (Value val : dropped)
    EXPECT_EQ(val, v);
}

TEST_F(RepeatedValueRangeTest, TakeFront) {
  Value v = makeValue();
  RepeatedValue rep{v};
  ValueRange range(rep, 8);

  ValueRange taken = range.take_front(3);
  EXPECT_EQ(taken.size(), 3u);
  for (Value val : taken)
    EXPECT_EQ(val, v);
}

TEST_F(RepeatedValueRangeTest, TakeBack) {
  Value v = makeValue();
  RepeatedValue rep{v};
  ValueRange range(rep, 8);

  ValueRange taken = range.take_back(2);
  EXPECT_EQ(taken.size(), 2u);
  for (Value val : taken)
    EXPECT_EQ(val, v);
}

//===----------------------------------------------------------------------===//
// llvm::Repeated<Value> implicit conversion to ValueRange
//===----------------------------------------------------------------------===//

TEST_F(RepeatedValueRangeTest, RepeatedImplicitConversion) {
  Value v = makeValue();
  llvm::Repeated<Value> rep(4, v);
  ValueRange range = rep;

  EXPECT_EQ(range.size(), 4u);
  for (Value val : range)
    EXPECT_EQ(val, v);
}

TEST_F(RepeatedValueRangeTest, RepeatedAsArgument) {
  Value v = makeValue();
  auto check = [&](ValueRange range) {
    EXPECT_EQ(range.size(), 3u);
    for (Value val : range)
      EXPECT_EQ(val, v);
  };
  check(llvm::Repeated<Value>(3, v));
}

TEST_F(RepeatedValueRangeTest, RepeatedSlice) {
  Value v = makeValue();
  llvm::Repeated<Value> rep(10, v);
  ValueRange range = rep;

  ValueRange sliced = range.slice(2, 5);
  EXPECT_EQ(sliced.size(), 5u);
  for (Value val : sliced)
    EXPECT_EQ(val, v);
}

} // namespace
