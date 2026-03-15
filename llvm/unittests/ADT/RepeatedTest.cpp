//===- RepeatedTest.cpp - Repeated unit tests -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Repeated.h"
#include "CountCopyAndMove.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace llvm;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::SizeIs;

//===----------------------------------------------------------------------===//
// Basic construction
//===----------------------------------------------------------------------===//

TEST(RepeatedTest, IntBasic) {
  Repeated<int> rep(5, 42);
  EXPECT_EQ(rep.front(), 42);
  EXPECT_THAT(rep, SizeIs(5));
}

TEST(RepeatedTest, StringCopy) {
  std::string s = "hello";
  Repeated<std::string> rep(3, s);
  EXPECT_EQ(rep.front(), "hello");
  EXPECT_THAT(rep, SizeIs(3));
  // Original should be unchanged (was copied).
  EXPECT_EQ(s, "hello");
}

TEST(RepeatedTest, StringMove) {
  std::string s = "hello";
  Repeated<std::string> rep(3, std::move(s));
  EXPECT_EQ(rep.front(), "hello");
  EXPECT_THAT(rep, SizeIs(3));
}

TEST(RepeatedTest, MoveOnly) {
  auto ptr = std::make_unique<int>(42);
  Repeated<std::unique_ptr<int>> rep(1, std::move(ptr));
  EXPECT_EQ(*rep.front(), 42);
  EXPECT_THAT(rep, SizeIs(1));
}

//===----------------------------------------------------------------------===//
// CTAD
//===----------------------------------------------------------------------===//

TEST(RepeatedTest, CTADInt) {
  auto rep = Repeated(3, 42);
  static_assert(std::is_same_v<decltype(rep), Repeated<int>>);
}

TEST(RepeatedTest, CTADString) {
  std::string s = "world";
  auto rep = Repeated(2, s);
  static_assert(std::is_same_v<decltype(rep), Repeated<std::string>>);
}

TEST(RepeatedTest, CTADFromLiteral) {
  // const char* deduced via std::decay_t in deduction guide.
  auto rep = Repeated(1, "literal");
  static_assert(std::is_same_v<decltype(rep), Repeated<const char *>>);
  EXPECT_STREQ(rep.front(), "literal");
}

//===----------------------------------------------------------------------===//
// Perfect forwarding -- copy/move counting
//===----------------------------------------------------------------------===//

TEST(RepeatedTest, ForwardingFromLvalue) {
  CountCopyAndMove::ResetCounts();
  CountCopyAndMove obj(7);
  CountCopyAndMove::ResetCounts();

  Repeated<CountCopyAndMove> rep(2, obj);

  EXPECT_EQ(rep.front().val, 7);
  EXPECT_EQ(CountCopyAndMove::CopyConstructions, 1);
  EXPECT_EQ(CountCopyAndMove::MoveConstructions, 0);
}

TEST(RepeatedTest, ForwardingFromRvalue) {
  CountCopyAndMove::ResetCounts();
  CountCopyAndMove obj(7);
  CountCopyAndMove::ResetCounts();

  Repeated<CountCopyAndMove> rep(2, std::move(obj));

  EXPECT_EQ(rep.front().val, 7);
  EXPECT_EQ(CountCopyAndMove::CopyConstructions, 0);
  EXPECT_EQ(CountCopyAndMove::MoveConstructions, 1);
}

//===----------------------------------------------------------------------===//
// Alignment
//===----------------------------------------------------------------------===//

TEST(RepeatedTest, StorageAlignment) {
  static_assert(alignof(RepeatedStorage<int>) >= alignof(void *));
  static_assert(alignof(RepeatedStorage<std::string>) >= alignof(void *));
  static_assert(alignof(RepeatedStorage<void *>) >= alignof(void *));
}

//===----------------------------------------------------------------------===//
// Range interface
//===----------------------------------------------------------------------===//

TEST(RepeatedTest, RangeBasic) {
  Repeated<int> rep(5, 42);
  EXPECT_THAT(rep, SizeIs(5));
  EXPECT_THAT(rep, Each(42));
  EXPECT_EQ(rep.front(), 42);
  EXPECT_EQ(rep.back(), 42);
  EXPECT_EQ(rep[0], 42);
  EXPECT_EQ(rep[4], 42);
}

TEST(RepeatedTest, RangeEmpty) {
  Repeated<int> rep(0, 7);
  EXPECT_THAT(rep, IsEmpty());
  EXPECT_EQ(rep.begin(), rep.end());
}

TEST(RepeatedTest, RangeForLoop) {
  Repeated<int> rep(4, 10);
  int sum = 0;
  for (int v : rep)
    sum += v;
  EXPECT_EQ(sum, 40);
}

//===----------------------------------------------------------------------===//
// Iterator properties
//===----------------------------------------------------------------------===//

TEST(RepeatedTest, IteratorRandomAccess) {
  Repeated<int> rep(10, 7);
  auto it = rep.begin();

  EXPECT_THAT(*it, Eq(7));
  EXPECT_THAT(*(it + 5), Eq(7));
  EXPECT_THAT(it[3], Eq(7));

  // Commutative addition: n + it.
  auto it3 = 5 + rep.begin();
  EXPECT_EQ(it3, rep.begin() + 5);

  auto it2 = it + 10;
  EXPECT_EQ(it2, rep.end());
  EXPECT_EQ(it2 - it, 10);
  EXPECT_EQ(rep.end() - rep.begin(), 10);

  EXPECT_LT(it, it2);
  EXPECT_LE(it, it);
  EXPECT_GT(it2, it);
  EXPECT_GE(it2, it2);
}

TEST(RepeatedTest, IteratorArrow) {
  Repeated<std::string> rep(3, "hello");
  auto it = rep.begin();
  EXPECT_EQ(it->size(), 5u);
}

TEST(RepeatedTest, IteratorDecrement) {
  Repeated<int> rep(5, 3);
  auto it = rep.end();
  --it;
  EXPECT_EQ(*it, 3);
  EXPECT_EQ(it, rep.begin() + 4);
}

TEST(RepeatedTest, IteratorCompoundAssignment) {
  Repeated<int> rep(10, 1);
  auto it = rep.begin();
  it += 7;
  EXPECT_EQ(it - rep.begin(), 7);
  it -= 3;
  EXPECT_EQ(it - rep.begin(), 4);
}

TEST(RepeatedTest, IteratorPostIncDec) {
  Repeated<int> rep(5, 99);
  auto it = rep.begin();
  auto old = it++;
  EXPECT_EQ(old, rep.begin());
  EXPECT_EQ(it, rep.begin() + 1);
  auto old2 = it--;
  EXPECT_EQ(old2, rep.begin() + 1);
  EXPECT_EQ(it, rep.begin());
}

//===----------------------------------------------------------------------===//
// STL algorithm compatibility
//===----------------------------------------------------------------------===//

TEST(RepeatedTest, StdCount) {
  Repeated<int> rep(100, 42);
  EXPECT_EQ(std::count(rep.begin(), rep.end(), 42), 100);
  EXPECT_EQ(std::count(rep.begin(), rep.end(), 0), 0);
}

TEST(RepeatedTest, StdAccumulate) {
  Repeated<int> rep(5, 10);
  EXPECT_EQ(std::accumulate(rep.begin(), rep.end(), 0), 50);
}

TEST(RepeatedTest, StdCopyToVector) {
  Repeated<std::string> rep(3, "abc");
  std::vector<std::string> out(rep.begin(), rep.end());
  EXPECT_THAT(out, ElementsAre("abc", "abc", "abc"));
}

TEST(RepeatedTest, StdDistance) {
  Repeated<int> rep(42, 0);
  EXPECT_EQ(std::distance(rep.begin(), rep.end()), 42);
}

//===----------------------------------------------------------------------===//
// Reverse iterators
//===----------------------------------------------------------------------===//

TEST(RepeatedTest, ReverseIterator) {
  Repeated<int> rep(5, 42);
  std::vector<int> reversed(rep.rbegin(), rep.rend());
  EXPECT_THAT(reversed, SizeIs(5));
  EXPECT_THAT(reversed, Each(42));
}

TEST(RepeatedTest, ReverseIteratorDistance) {
  Repeated<int> rep(7, 3);
  EXPECT_EQ(std::distance(rep.rbegin(), rep.rend()), 7);
}

TEST(RepeatedTest, ReverseIteratorEmpty) {
  Repeated<int> rep(0, 1);
  EXPECT_EQ(rep.rbegin(), rep.rend());
}

//===----------------------------------------------------------------------===//
// Iterator traits
//===----------------------------------------------------------------------===//

TEST(RepeatedTest, IteratorTraits) {
  using It = RepeatedIterator<int>;
  static_assert(std::is_same_v<std::iterator_traits<It>::iterator_category,
                               std::random_access_iterator_tag>);
  static_assert(std::is_same_v<std::iterator_traits<It>::value_type, int>);
  static_assert(
      std::is_same_v<std::iterator_traits<It>::difference_type, ptrdiff_t>);
}

//===----------------------------------------------------------------------===//
// Death tests (debug builds only)
//===----------------------------------------------------------------------===//

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
TEST(RepeatedTest, FrontOnEmptyDeath) {
  Repeated<int> rep(0, 42);
  EXPECT_DEATH(rep.front(), "front");
}

TEST(RepeatedTest, BackOnEmptyDeath) {
  Repeated<int> rep(0, 42);
  EXPECT_DEATH(rep.back(), "back");
}

TEST(RepeatedTest, IndexOutOfBoundsDeath) {
  Repeated<int> rep(3, 42);
  EXPECT_DEATH(rep[3], "index out of bounds");
}
#endif
