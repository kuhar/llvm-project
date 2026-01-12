//===- STLForwardCompatTest.cpp - Unit tests for STLForwardCompat ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLForwardCompat.h"
#include "CountCopyAndMove.h"
#include "gtest/gtest.h"

#include <optional>
#include <type_traits>
#include <utility>

using namespace llvm;

namespace {

template <typename T>
class STLForwardCompatRemoveCVRefTest : public ::testing::Test {};

using STLForwardCompatRemoveCVRefTestTypes = ::testing::Types<
    // clang-format off
    std::pair<int, int>,
    std::pair<int &, int>,
    std::pair<const int, int>,
    std::pair<volatile int, int>,
    std::pair<const volatile int &, int>,
    std::pair<int *, int *>,
    std::pair<int *const, int *>,
    std::pair<const int *, const int *>,
    std::pair<int *&, int *>
    // clang-format on
    >;

TYPED_TEST_SUITE(STLForwardCompatRemoveCVRefTest,
                 STLForwardCompatRemoveCVRefTestTypes, );

TYPED_TEST(STLForwardCompatRemoveCVRefTest, RemoveCVRef) {
  using From = typename TypeParam::first_type;
  using To = typename TypeParam::second_type;
  EXPECT_TRUE(
      (std::is_same<typename llvm::remove_cvref<From>::type, To>::value));
}

TYPED_TEST(STLForwardCompatRemoveCVRefTest, RemoveCVRefT) {
  using From = typename TypeParam::first_type;
  EXPECT_TRUE((std::is_same<typename llvm::remove_cvref<From>::type,
                            llvm::remove_cvref_t<From>>::value));
}

template <typename T> class TypeIdentityTest : public ::testing::Test {
public:
  using TypeIdentity = llvm::type_identity<T>;
};

struct A {
  struct B {};
};
using TypeIdentityTestTypes =
    ::testing::Types<int, volatile int, A, const A::B>;

TYPED_TEST_SUITE(TypeIdentityTest, TypeIdentityTestTypes, /*NameGenerator*/);

TYPED_TEST(TypeIdentityTest, Identity) {
  // TestFixture is the instantiated TypeIdentityTest.
  EXPECT_TRUE(
      (std::is_same_v<TypeParam, typename TestFixture::TypeIdentity::type>));
}

TEST(TransformTest, TransformStd) {
  std::optional<int> A;

  std::optional<int> B = llvm::transformOptional(A, [&](int N) { return N + 1; });
  EXPECT_FALSE(B.has_value());

  A = 3;
  std::optional<int> C = llvm::transformOptional(A, [&](int N) { return N + 1; });
  EXPECT_TRUE(C.has_value());
  EXPECT_EQ(4, *C);
}

TEST(TransformTest, MoveTransformStd) {
  using llvm::CountCopyAndMove;

  std::optional<CountCopyAndMove> A;

  CountCopyAndMove::ResetCounts();
  std::optional<int> B = llvm::transformOptional(
      std::move(A), [&](const CountCopyAndMove &M) { return M.val + 2; });
  EXPECT_FALSE(B.has_value());
  EXPECT_EQ(0, CountCopyAndMove::TotalCopies());
  EXPECT_EQ(0, CountCopyAndMove::MoveConstructions);
  EXPECT_EQ(0, CountCopyAndMove::MoveAssignments);
  EXPECT_EQ(0, CountCopyAndMove::Destructions);

  A = CountCopyAndMove(5);
  CountCopyAndMove::ResetCounts();
  std::optional<int> C = llvm::transformOptional(
      std::move(A), [&](const CountCopyAndMove &M) { return M.val + 2; });
  EXPECT_TRUE(C.has_value());
  EXPECT_EQ(7, *C);
  EXPECT_EQ(0, CountCopyAndMove::TotalCopies());
  EXPECT_EQ(0, CountCopyAndMove::MoveConstructions);
  EXPECT_EQ(0, CountCopyAndMove::MoveAssignments);
  EXPECT_EQ(0, CountCopyAndMove::Destructions);
}

TEST(TransformTest, TransformLlvm) {
  std::optional<int> A;

  std::optional<int> B =
      llvm::transformOptional(A, [&](int N) { return N + 1; });
  EXPECT_FALSE(B.has_value());

  A = 3;
  std::optional<int> C =
      llvm::transformOptional(A, [&](int N) { return N + 1; });
  EXPECT_TRUE(C.has_value());
  EXPECT_EQ(4, *C);
}

TEST(TransformTest, MoveTransformLlvm) {
  using llvm::CountCopyAndMove;

  std::optional<CountCopyAndMove> A;

  CountCopyAndMove::ResetCounts();
  std::optional<int> B = llvm::transformOptional(
      std::move(A), [&](const CountCopyAndMove &M) { return M.val + 2; });
  EXPECT_FALSE(B.has_value());
  EXPECT_EQ(0, CountCopyAndMove::TotalCopies());
  EXPECT_EQ(0, CountCopyAndMove::MoveConstructions);
  EXPECT_EQ(0, CountCopyAndMove::MoveAssignments);
  EXPECT_EQ(0, CountCopyAndMove::Destructions);

  A = CountCopyAndMove(5);
  CountCopyAndMove::ResetCounts();
  std::optional<int> C = llvm::transformOptional(
      std::move(A), [&](const CountCopyAndMove &M) { return M.val + 2; });
  EXPECT_TRUE(C.has_value());
  EXPECT_EQ(7, *C);
  EXPECT_EQ(0, CountCopyAndMove::TotalCopies());
  EXPECT_EQ(0, CountCopyAndMove::MoveConstructions);
  EXPECT_EQ(0, CountCopyAndMove::MoveAssignments);
  EXPECT_EQ(0, CountCopyAndMove::Destructions);
}

TEST(TransformTest, TransformCategory) {
  struct StructA {
    int x;
  };
  struct StructB : StructA {
    StructB(StructA &&A) : StructA(std::move(A)) {}
  };

  std::optional<StructA> A{StructA{}};
  llvm::transformOptional(A, [](auto &&s) {
    EXPECT_FALSE(std::is_rvalue_reference_v<decltype(s)>);
    return StructB{std::move(s)};
  });

  llvm::transformOptional(std::move(A), [](auto &&s) {
    EXPECT_TRUE(std::is_rvalue_reference_v<decltype(s)>);
    return StructB{std::move(s)};
  });
}

TEST(TransformTest, ToUnderlying) {
  enum E { A1 = 0, B1 = -1 };
  static_assert(llvm::to_underlying(A1) == 0);
  static_assert(llvm::to_underlying(B1) == -1);

  enum E2 : unsigned char { A2 = 0, B2 };
  static_assert(
      std::is_same_v<unsigned char, decltype(llvm::to_underlying(A2))>);
  static_assert(llvm::to_underlying(A2) == 0);
  static_assert(llvm::to_underlying(B2) == 1);

  enum class E3 { A3 = -1, B3 };
  static_assert(std::is_same_v<int, decltype(llvm::to_underlying(E3::A3))>);
  static_assert(llvm::to_underlying(E3::A3) == -1);
  static_assert(llvm::to_underlying(E3::B3) == 0);
}

TEST(STLForwardCompatTest, IdentityCxx20) {
  llvm::identity identity;

  // Test with an lvalue.
  int X = 42;
  int &Y = identity(X);
  EXPECT_EQ(&X, &Y);

  // Test with a const lvalue.
  const int CX = 10;
  const int &CY = identity(CX);
  EXPECT_EQ(&CX, &CY);

  // Test with an rvalue.
  EXPECT_EQ(identity(123), 123);

  // Test perfect forwarding.
  static_assert(std::is_same_v<int &, decltype(identity(X))>);
  static_assert(std::is_same_v<const int &, decltype(identity(CX))>);
  static_assert(std::is_same_v<int &&, decltype(identity(int(5)))>);
}

TEST(STLForwardCompatTest, BindFrontReferences) {
  // All bound arguments are forwarded (for ints, this is a copy) into the
  // wrapper. Call arguments are forwarded with their original value category.
  int A = 1;
  const int B = 2;
  int C = 3;
  int D = 4;
  const int E = 5;
  int F = 6;

  auto TestTypes = [](auto &&AArg, auto &&BArg, auto &&CArg, auto &&DArg,
                      auto &&EArg, auto &&FArg) {
    // Bound args: all stored as values, passed as lvalue refs.
    EXPECT_EQ(AArg, 1);
    static_assert(std::is_same_v<decltype(AArg), int &>);
    EXPECT_EQ(BArg, 2);
    static_assert(std::is_same_v<decltype(BArg), int &>); // Const decayed away.
    EXPECT_EQ(CArg, 3);
    static_assert(std::is_same_v<decltype(CArg), int &>);
    // Call args: forwarded with original value category.
    EXPECT_EQ(DArg, 4);
    static_assert(std::is_same_v<decltype(DArg), int &>);
    EXPECT_EQ(EArg, 5);
    static_assert(std::is_same_v<decltype(EArg), const int &>);
    EXPECT_EQ(FArg, 6);
    static_assert(std::is_same_v<decltype(FArg), int &&>);

    ++DArg;
  };

  bind_front(TestTypes, A, B, std::move(C))(D, E, std::move(F));
  EXPECT_EQ(A, 1); // A was copied, original unchanged.
  EXPECT_EQ(D, 5); // D was passed by reference and incremented.
}

TEST(STLForwardCompatTest, BindBackReferences) {
  // With std::decay_t, all bound arguments are copied into the wrapper.
  // Call arguments are forwarded with their original value category.
  int A = 1;
  const int B = 2;
  int C = 3;
  int D = 4;
  const int E = 5;
  int F = 6;

  auto TestTypes = [](auto &&AArg, auto &&BArg, auto &&CArg, auto &&DArg,
                      auto &&EArg, auto &&FArg) {
    // Call args: forwarded with original value category.
    EXPECT_EQ(AArg, 1);
    static_assert(std::is_same_v<decltype(AArg), int &>);
    EXPECT_EQ(BArg, 2);
    static_assert(std::is_same_v<decltype(BArg), const int &>);
    EXPECT_EQ(CArg, 3);
    static_assert(std::is_same_v<decltype(CArg), int &&>);
    // Bound args: all stored as values, passed as lvalue refs.
    EXPECT_EQ(DArg, 4);
    static_assert(std::is_same_v<decltype(DArg), int &>);
    EXPECT_EQ(EArg, 5);
    static_assert(std::is_same_v<decltype(EArg), int &>); // Const decayed away.
    EXPECT_EQ(FArg, 6);
    static_assert(std::is_same_v<decltype(FArg), int &>);

    ++AArg;
  };

  bind_back(TestTypes, D, E, std::move(F))(A, B, std::move(C));
  EXPECT_EQ(A, 2); // A was passed by reference and incremented.
  EXPECT_EQ(D, 4); // D was copied, original unchanged.
}

TEST(STLForwardCompatTest, BindFrontMutableStorage) {
  // With std::decay_t, A is copied into the wrapper. The stored copy can be
  // mutated across calls, but the original A is unchanged.
  int A = 1;

  auto TestMutation = [](int &AArg, int &BArg, auto ExtraCheckFn) {
    ++AArg;
    ++BArg;
    ExtraCheckFn(AArg, BArg);
  };

  auto BoundA = bind_front(TestMutation, A, 42);
  BoundA([](int AVal, int BVal) {
    EXPECT_EQ(AVal, 2); // Stored copy incremented from 1.
    EXPECT_EQ(BVal, 43);
  });
  EXPECT_EQ(A, 1); // Original unchanged.

  BoundA([](int AVal, int BVal) {
    EXPECT_EQ(AVal, 3); // Stored copy incremented again.
    EXPECT_EQ(BVal, 44);
  });
  EXPECT_EQ(A, 1); // Original still unchanged.
}

TEST(STLForwardCompatTest, BindBackMutableStorage) {
  // With std::decay_t, A is copied into the wrapper. The stored copy can be
  // mutated across calls, but the original A is unchanged.
  int A = 1;

  auto TestMutation = [](auto ExtraCheckFn, int &AArg, int &BArg) {
    ++AArg;
    ++BArg;
    ExtraCheckFn(AArg, BArg);
  };

  auto BoundA = bind_back(TestMutation, A, 42);
  BoundA([](int AVal, int BVal) {
    EXPECT_EQ(AVal, 2); // Stored copy incremented from 1.
    EXPECT_EQ(BVal, 43);
  });
  EXPECT_EQ(A, 1); // Original unchanged.

  BoundA([](int AVal, int BVal) {
    EXPECT_EQ(AVal, 3); // Stored copy incremented again.
    EXPECT_EQ(BVal, 44);
  });
  EXPECT_EQ(A, 1); // Original still unchanged.
}

TEST(STLForwardCompatTest, BindFrontBindBackConstexpr) {
  static constexpr auto Fn1 = bind_front([](int A, int B) { return A + B; }, 1);
  static_assert(Fn1(3) == 4);
  static constexpr auto Fn2 = bind_back([](int A, int B) { return A + B; }, 1);
  static_assert(Fn2(3) == 4);
}

// Use std::ref/std::cref to bind references (bound args are decay-copied).
TEST(STLForwardCompatTest, BindWithReferenceWrapper) {
  int X = 1;
  auto Increment = bind_front([](int &Val) { ++Val; }, std::ref(X));
  Increment();
  EXPECT_EQ(X, 2);
  Increment();
  EXPECT_EQ(X, 3);
}

TEST(STLForwardCompat, BindFrontBindBack) {
  std::vector<int> V;
  auto MulAdd = [](int A, int B, int C) { return A * (B + C) == 12; };
  auto MulAdd1 = [](const int &A, const int &B, const int &C) {
    return A * (B + C) == 12;
  };
  auto Mul0 = bind_back(MulAdd, 4, 2);
  auto MulL = bind_front(MulAdd1, 2, 4);
  auto Mul20 = bind_back(MulAdd, 4);
  auto Mul21 = bind_front(MulAdd1, 2);
  EXPECT_TRUE(all_of(V, Mul0));
  EXPECT_TRUE(all_of(V, MulL));

  V.push_back(2);
  EXPECT_TRUE(all_of(V, Mul0));
  EXPECT_TRUE(all_of(V, MulL));

  V.push_back(2);
  V.push_back(2);
  EXPECT_TRUE(all_of(V, Mul0));
  EXPECT_TRUE(all_of(V, MulL));

  auto Spec0 = bind_front(Mul20, 2);
  auto Spec1 = bind_back(Mul21, 4);
  EXPECT_TRUE(all_of(V, Spec0));
  EXPECT_TRUE(all_of(V, Spec1));

  V.push_back(3);
  EXPECT_FALSE(all_of(V, Mul0));
  EXPECT_FALSE(all_of(V, MulL));
  EXPECT_FALSE(all_of(V, Spec0));
  EXPECT_FALSE(all_of(V, Spec1));
  EXPECT_TRUE(any_of(V, Spec0));
  EXPECT_TRUE(any_of(V, Spec1));
}

} // namespace
