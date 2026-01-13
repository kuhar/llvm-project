//===- llvm/unittest/ADT/RandomAccessDeque.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RandomAccessDeque unit tests.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/MathExtras.h"
#include "gtest/gtest.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <new>
#include <type_traits>
#include <utility>

namespace llvm {
namespace {

/// Calculate the largest power-of-2 bucket size such that one bucket
/// fits within PageSize bytes. Returns at least 1.
///
/// Ex. defaultBucketSize<int>() == 1024 (for 4KB pages)
/// Ex. defaultBucketSize<char[256]>() == 16
template <typename T, size_t PageSize = 4096>
constexpr uint32_t defaultBucketSize() {
  size_t MaxElements = PageSize / sizeof(T);
  if (MaxElements == 0)
    return 1;
  // bit_floor: largest power of 2 <= MaxElements
  return static_cast<uint32_t>(size_t{1}
                               << (bit_width_constexpr(MaxElements) - 1));
}

/// A container with pointer stability, random access, and exponentially
/// growing buckets. Supports only push_back/emplace_back and pop_back.
///
/// Buckets grow exponentially: sizes are N, 2N, 4N, 8N, ...
/// - Bucket 0: indices [0, N)           size N
/// - Bucket 1: indices [N, 3N)          size 2N
/// - Bucket k: indices [N*(2^k-1), N*(2^(k+1)-1))  size N*2^k
///
/// Template parameters:
///   T - element type
///   InitialBucketSize - initial bucket size (must be power of 2, defaults to
///                       largest power of 2 that fits in one page)
template <typename T, uint32_t InitialBucketSize = defaultBucketSize<T>()>
class RandomAccessDeque {
  static_assert(InitialBucketSize != 0 &&
                    (InitialBucketSize & (InitialBucketSize - 1)) == 0,
                "InitialBucketSize must be a power of 2");
  static constexpr uint32_t N = InitialBucketSize;
  static constexpr uint32_t Log2N = ConstantLog2<N>();
  static constexpr uint32_t MaxBuckets = 15;

  T *Buckets[MaxBuckets] = {};
  uint32_t Size = 0;
  uint32_t NumBuckets = 0;

  /// Get bucket index and offset within bucket for a given element index.
  static std::pair<uint32_t, uint32_t> getBucketAndOffset(size_t Index) {
    size_t Adjusted = Index + N;
    uint32_t MsbPos = llvm::bit_width(Adjusted) - 1;
    uint32_t Bucket = MsbPos - Log2N;
    uint32_t Offset = Adjusted ^ (size_t{1} << MsbPos);
    return {Bucket, Offset};
  }

  /// Get the size of bucket K (N * 2^K).
  static constexpr uint32_t getBucketSize(uint32_t K) { return N << K; }

  /// Get starting element index of bucket K.
  static constexpr uint32_t getBucketStart(uint32_t K) {
    return K == 0 ? 0 : N * ((uint32_t{1} << K) - 1);
  }

  /// Get number of elements in bucket K given total Size elements.
  static uint32_t getElementsInBucket(uint32_t K, uint32_t TotalSize) {
    uint32_t Start = getBucketStart(K);
    if (TotalSize <= Start)
      return 0;
    return std::min(TotalSize - Start, getBucketSize(K));
  }

  /// Allocate a new bucket.
  void allocateBucket(uint32_t BucketIdx) {
    assert(BucketIdx < MaxBuckets && "Bucket index out of range");
    assert(!Buckets[BucketIdx] && "Bucket already allocated");
    uint32_t BucketSize = getBucketSize(BucketIdx);
    Buckets[BucketIdx] =
        static_cast<T *>(::operator new(sizeof(T) * BucketSize));
    NumBuckets = BucketIdx + 1;
  }

  /// Destroy all elements and deallocate buckets.
  void destroyAndDeallocate() {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      for (uint32_t I = 0; I < Size; ++I) {
        auto [Bucket, Offset] = getBucketAndOffset(I);
        Buckets[Bucket][Offset].~T();
      }
    }
    for (uint32_t I = 0; I < NumBuckets; ++I) {
      ::operator delete(Buckets[I]);
      Buckets[I] = nullptr;
    }
    Size = 0;
    NumBuckets = 0;
  }

public:
  using value_type = T;
  using size_type = uint32_t;
  using difference_type = std::ptrdiff_t;
  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using const_pointer = const T *;

  //===--------------------------------------------------------------------===//
  // Iterator
  //===--------------------------------------------------------------------===//

  template <bool IsConst>
  class IteratorImpl
      : public iterator_facade_base<
            IteratorImpl<IsConst>, std::random_access_iterator_tag, T,
            std::ptrdiff_t, std::conditional_t<IsConst, const T *, T *>,
            std::conditional_t<IsConst, const T &, T &>> {
    using BaseT =
        iterator_facade_base<IteratorImpl<IsConst>,
                             std::random_access_iterator_tag, T, std::ptrdiff_t,
                             std::conditional_t<IsConst, const T *, T *>,
                             std::conditional_t<IsConst, const T &, T &>>;
    using ContainerPtr = std::conditional_t<IsConst, const RandomAccessDeque *,
                                            RandomAccessDeque *>;
    ContainerPtr Container = nullptr;
    uint32_t Index = 0;
    template <bool> friend class IteratorImpl;

  public:
    IteratorImpl() = default;
    IteratorImpl(ContainerPtr C, uint32_t I) : Container(C), Index(I) {}

    // Allow conversion from non-const to const iterator.
    template <bool WasConst, typename = std::enable_if_t<IsConst && !WasConst>>
    IteratorImpl(const IteratorImpl<WasConst> &Other)
        : Container(Other.Container), Index(Other.Index) {}

    std::conditional_t<IsConst, const T &, T &> operator*() const {
      return (*Container)[Index];
    }

    bool operator==(const IteratorImpl &RHS) const {
      return Index == RHS.Index;
    }

    bool operator<(const IteratorImpl &RHS) const { return Index < RHS.Index; }

    // Bring base class operator- (iter - n) into scope alongside our
    // difference.
    using BaseT::operator-;
    std::ptrdiff_t operator-(const IteratorImpl &RHS) const {
      return static_cast<std::ptrdiff_t>(Index) -
             static_cast<std::ptrdiff_t>(RHS.Index);
    }

    IteratorImpl &operator+=(std::ptrdiff_t N) {
      Index += N;
      return *this;
    }

    IteratorImpl &operator-=(std::ptrdiff_t N) {
      Index -= N;
      return *this;
    }
  };

  using iterator = IteratorImpl<false>;
  using const_iterator = IteratorImpl<true>;

  //===--------------------------------------------------------------------===//
  // Constructors / Destructor
  //===--------------------------------------------------------------------===//

  RandomAccessDeque() = default;

  RandomAccessDeque(const RandomAccessDeque &Other) {
    reserve(Other.Size);
    if constexpr (std::is_trivially_copyable_v<T>) {
      for (uint32_t I = 0; I < NumBuckets; ++I) {
        uint32_t Count = getElementsInBucket(I, Other.Size);
        std::memcpy(Buckets[I], Other.Buckets[I], sizeof(T) * Count);
      }
      Size = Other.Size;
    } else {
      for (const auto &Elem : Other)
        push_back(Elem);
    }
  }

  RandomAccessDeque(RandomAccessDeque &&Other) {
    llvm::copy(Other.Buckets, std::begin(Buckets));
    Size = Other.Size;
    NumBuckets = Other.NumBuckets;
    llvm::fill(Other.Buckets, nullptr);
    Other.Size = 0;
    Other.NumBuckets = 0;
  }

  RandomAccessDeque &operator=(const RandomAccessDeque &Other) {
    if (this != &Other) {
      clear();
      reserve(Other.Size);
      if constexpr (std::is_trivially_copyable_v<T>) {
        for (uint32_t I = 0; I < NumBuckets; ++I) {
          uint32_t Count = getElementsInBucket(I, Other.Size);
          std::memcpy(Buckets[I], Other.Buckets[I], sizeof(T) * Count);
        }
        Size = Other.Size;
      } else {
        for (const auto &Elem : Other)
          push_back(Elem);
      }
    }
    return *this;
  }

  RandomAccessDeque &operator=(RandomAccessDeque &&Other) {
    if (this != &Other) {
      destroyAndDeallocate();
      llvm::copy(Other.Buckets, Buckets);
      Size = Other.Size;
      NumBuckets = Other.NumBuckets;
      llvm::fill(Other.Buckets, nullptr);
      Other.Size = 0;
      Other.NumBuckets = 0;
    }
    return *this;
  }

  ~RandomAccessDeque() { destroyAndDeallocate(); }

  //===--------------------------------------------------------------------===//
  // Capacity
  //===--------------------------------------------------------------------===//

  [[nodiscard]] bool empty() const { return Size == 0; }
  [[nodiscard]] uint32_t size() const { return Size; }

  /// Maximum number of elements the container can hold.
  [[nodiscard]] static constexpr uint32_t max_size() {
    // N * (2^MaxBuckets - 1)
    return N * ((uint32_t{1} << MaxBuckets) - 1);
  }

  /// Reserve space for at least N elements by pre-allocating buckets.
  void reserve(uint32_t NewCapacity) {
    while (capacity() < NewCapacity) {
      allocateBucket(NumBuckets);
    }
  }

  /// Current capacity (sum of all allocated bucket sizes).
  [[nodiscard]] uint32_t capacity() const {
    if (NumBuckets == 0)
      return 0;
    // Sum of geometric series: N * (2^NumBuckets - 1)
    return N * ((uint32_t{1} << NumBuckets) - 1);
  }

  //===--------------------------------------------------------------------===//
  // Element Access
  //===--------------------------------------------------------------------===//

  /// Hint for accelerating nearby element accesses.
  /// Users can pass this to operator[] to avoid recomputing bucket/offset
  /// when accessing elements sequentially or within the same bucket.
  struct IndexHint {
    uint32_t LastIndex = ~0u; // ~0u means invalid/uninitialized
    uint32_t BucketIdx = 0;
    uint32_t Offset = 0;
  };

  [[nodiscard]] reference operator[](uint32_t Index) {
    assert(Index < Size && "Index out of bounds");
    auto [Bucket, Offset] = getBucketAndOffset(Index);
    return Buckets[Bucket][Offset];
  }

  [[nodiscard]] const_reference operator[](uint32_t Index) const {
    assert(Index < Size && "Index out of bounds");
    auto [Bucket, Offset] = getBucketAndOffset(Index);
    return Buckets[Bucket][Offset];
  }

  /// Access element with hint. If Index is in the same bucket as the hint,
  /// avoids recomputing bucket/offset. Updates the hint for next access.
  [[nodiscard]] reference at(uint32_t Index, IndexHint &Hint) {
    assert(Index < Size && "Index out of bounds");
    updateHint(Index, Hint);
    return Buckets[Hint.BucketIdx][Hint.Offset];
  }

  [[nodiscard]] const_reference at(uint32_t Index, IndexHint &Hint) const {
    assert(Index < Size && "Index out of bounds");
    updateHint(Index, Hint);
    return Buckets[Hint.BucketIdx][Hint.Offset];
  }

private:
  /// Update hint for a new index. If the new index is in the same bucket
  /// as the hint, just update the offset. Otherwise, recompute everything.
  void updateHint(uint32_t Index, IndexHint &Hint) const {
    if (Hint.LastIndex != ~0u) {
      // Check if we can stay in the same bucket
      int64_t Delta = static_cast<int64_t>(Index) - Hint.LastIndex;
      int64_t NewOffset = static_cast<int64_t>(Hint.Offset) + Delta;
      if (NewOffset >= 0 &&
          NewOffset < static_cast<int64_t>(getBucketSize(Hint.BucketIdx))) {
        // Same bucket - just update offset
        Hint.Offset = static_cast<uint32_t>(NewOffset);
        Hint.LastIndex = Index;
        return;
      }
    }
    // Fall back to full computation
    auto [BucketIdx, Offset] = getBucketAndOffset(Index);
    Hint.LastIndex = Index;
    Hint.BucketIdx = BucketIdx;
    Hint.Offset = Offset;
  }

public:
  [[nodiscard]] reference front() {
    assert(!empty() && "front() on empty container");
    return Buckets[0][0];
  }

  [[nodiscard]] const_reference front() const {
    assert(!empty() && "front() on empty container");
    return Buckets[0][0];
  }

  [[nodiscard]] reference back() {
    assert(!empty() && "back() on empty container");
    return (*this)[Size - 1];
  }

  [[nodiscard]] const_reference back() const {
    assert(!empty() && "back() on empty container");
    return (*this)[Size - 1];
  }

  //===--------------------------------------------------------------------===//
  // Modifiers
  //===--------------------------------------------------------------------===//

  void push_back(const T &Value) { emplace_back(Value); }
  void push_back(T &&Value) { emplace_back(std::move(Value)); }

  template <typename... Args> reference emplace_back(Args &&...args) {
    auto [Bucket, Offset] = getBucketAndOffset(Size);
    if (Bucket >= NumBuckets)
      allocateBucket(Bucket);
    T *Ptr = Buckets[Bucket] + Offset;
    ::new (static_cast<void *>(Ptr)) T(std::forward<Args>(args)...);
    ++Size;
    return *Ptr;
  }

  void pop_back() {
    assert(!empty() && "pop_back() on empty container");
    --Size;
    if constexpr (!std::is_trivially_destructible_v<T>) {
      auto [Bucket, Offset] = getBucketAndOffset(Size);
      Buckets[Bucket][Offset].~T();
    }
  }

  [[nodiscard]] T pop_back_val() {
    assert(!empty() && "pop_back_val() on empty container");
    T Result = std::move(back());
    pop_back();
    return Result;
  }

  void clear() {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      for (uint32_t I = 0; I < Size; ++I) {
        auto [Bucket, Offset] = getBucketAndOffset(I);
        Buckets[Bucket][Offset].~T();
      }
    }
    Size = 0;
    // Keep buckets allocated for potential reuse.
  }

  //===--------------------------------------------------------------------===//
  // Iterators
  //===--------------------------------------------------------------------===//

  [[nodiscard]] iterator begin() { return iterator(this, 0); }
  [[nodiscard]] const_iterator begin() const { return const_iterator(this, 0); }
  [[nodiscard]] const_iterator cbegin() const {
    return const_iterator(this, 0);
  }

  [[nodiscard]] iterator end() { return iterator(this, Size); }
  [[nodiscard]] const_iterator end() const {
    return const_iterator(this, Size);
  }
  [[nodiscard]] const_iterator cend() const {
    return const_iterator(this, Size);
  }
};

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

TEST(RandomAccessDeque, BasicOperations) {
  RandomAccessDeque<int, 4> D;

  EXPECT_TRUE(D.empty());
  EXPECT_EQ(D.size(), 0u);

  // Push some elements
  D.push_back(10);
  D.push_back(20);
  D.push_back(30);

  EXPECT_FALSE(D.empty());
  EXPECT_EQ(D.size(), 3u);
  EXPECT_EQ(D[0], 10);
  EXPECT_EQ(D[1], 20);
  EXPECT_EQ(D[2], 30);
  EXPECT_EQ(D.front(), 10);
  EXPECT_EQ(D.back(), 30);
}

TEST(RandomAccessDeque, GrowAcrossBuckets) {
  RandomAccessDeque<int, 4> D;

  // Fill bucket 0 (size 4) and bucket 1 (size 8) = 12 elements
  for (int I = 0; I < 12; ++I)
    D.push_back(I * 10);

  EXPECT_EQ(D.size(), 12u);

  // Verify all elements
  for (uint32_t I = 0; I < D.size(); ++I)
    EXPECT_EQ(D[I], static_cast<int>(I * 10));
}

TEST(RandomAccessDeque, PointerStability) {
  RandomAccessDeque<int, 4> D;

  D.push_back(100);
  int *Ptr0 = &D[0];

  // Add many more elements, causing new bucket allocations
  for (int I = 1; I < 100; ++I)
    D.push_back(I);

  // Original pointer should still be valid
  EXPECT_EQ(*Ptr0, 100);
  EXPECT_EQ(&D[0], Ptr0);
}

TEST(RandomAccessDeque, Iterator) {
  RandomAccessDeque<int, 4> D;
  for (int I = 0; I < 20; ++I)
    D.push_back(I);

  // Test forward iteration
  int Expected = 0;
  for (auto It = D.begin(); It != D.end(); ++It) {
    EXPECT_EQ(*It, Expected);
    ++Expected;
  }
  EXPECT_EQ(Expected, 20);

  // Test range-based for
  Expected = 0;
  for (int Val : D) {
    EXPECT_EQ(Val, Expected);
    ++Expected;
  }
}

TEST(RandomAccessDeque, RandomAccessIterator) {
  RandomAccessDeque<int, 4> D;
  for (int I = 0; I < 20; ++I)
    D.push_back(I * 2);

  auto It = D.begin();

  // Test operator[]
  EXPECT_EQ(It[0], 0);
  EXPECT_EQ(It[5], 10);
  EXPECT_EQ(It[19], 38);

  // Test arithmetic
  EXPECT_EQ(*(It + 3), 6);
  EXPECT_EQ(*(D.end() - 1), 38);

  // Test comparison
  EXPECT_TRUE(D.begin() < D.end());
  EXPECT_TRUE(D.begin() <= D.begin());
  EXPECT_TRUE(D.end() > D.begin());
  EXPECT_EQ(D.end() - D.begin(), 20);
}

TEST(RandomAccessDeque, IndexHint) {
  RandomAccessDeque<int, 4> D;
  // Fill multiple buckets: bucket 0 (4), bucket 1 (8), bucket 2 (16) = 28 total
  for (int I = 0; I < 28; ++I)
    D.push_back(I * 10);

  RandomAccessDeque<int, 4>::IndexHint Hint;

  // First access initializes the hint
  EXPECT_EQ(D.at(5, Hint), 50);
  EXPECT_EQ(Hint.LastIndex, 5u);

  // Sequential forward access within same bucket (bucket 1: indices 4-11)
  EXPECT_EQ(D.at(6, Hint), 60);
  EXPECT_EQ(D.at(7, Hint), 70);
  EXPECT_EQ(D.at(8, Hint), 80);

  // Backward access within same bucket
  EXPECT_EQ(D.at(5, Hint), 50);

  // Jump to different bucket (bucket 2: indices 12-27)
  EXPECT_EQ(D.at(15, Hint), 150);
  EXPECT_EQ(Hint.BucketIdx, 2u);

  // Sequential access in new bucket
  EXPECT_EQ(D.at(16, Hint), 160);
  EXPECT_EQ(D.at(17, Hint), 170);

  // Jump back to bucket 0
  EXPECT_EQ(D.at(0, Hint), 0);
  EXPECT_EQ(Hint.BucketIdx, 0u);

  // Const access with hint
  const auto &CD = D;
  RandomAccessDeque<int, 4>::IndexHint ConstHint;
  EXPECT_EQ(CD.at(10, ConstHint), 100);
  EXPECT_EQ(CD.at(11, ConstHint), 110);
}

TEST(RandomAccessDeque, PopBack) {
  RandomAccessDeque<int, 4> D;
  for (int I = 0; I < 10; ++I)
    D.push_back(I);

  EXPECT_EQ(D.size(), 10u);
  EXPECT_EQ(D.back(), 9);

  D.pop_back();
  EXPECT_EQ(D.size(), 9u);
  EXPECT_EQ(D.back(), 8);

  D.pop_back();
  D.pop_back();
  EXPECT_EQ(D.size(), 7u);
  EXPECT_EQ(D.back(), 6);
}

TEST(RandomAccessDeque, Clear) {
  RandomAccessDeque<int, 4> D;
  for (int I = 0; I < 20; ++I)
    D.push_back(I);

  EXPECT_EQ(D.size(), 20u);
  EXPECT_GT(D.capacity(), 0u);

  D.clear();
  EXPECT_EQ(D.size(), 0u);
  EXPECT_TRUE(D.empty());
  // Capacity is retained after clear
  EXPECT_GT(D.capacity(), 0u);

  // Can still push after clear
  D.push_back(42);
  EXPECT_EQ(D[0], 42);
}

TEST(RandomAccessDeque, CopyConstruct) {
  RandomAccessDeque<int, 4> D1;
  for (int I = 0; I < 15; ++I)
    D1.push_back(I);

  RandomAccessDeque<int, 4> D2(D1);

  EXPECT_EQ(D2.size(), D1.size());
  for (uint32_t I = 0; I < D1.size(); ++I)
    EXPECT_EQ(D2[I], D1[I]);

  // Modifications to D2 don't affect D1
  D2[0] = 999;
  EXPECT_EQ(D1[0], 0);
  EXPECT_EQ(D2[0], 999);
}

TEST(RandomAccessDeque, MoveConstruct) {
  RandomAccessDeque<int, 4> D1;
  for (int I = 0; I < 15; ++I)
    D1.push_back(I);

  int *OrigPtr = &D1[0];
  RandomAccessDeque<int, 4> D2(std::move(D1));

  EXPECT_EQ(D2.size(), 15u);
  EXPECT_EQ(&D2[0], OrigPtr); // Pointer stability across move
  EXPECT_TRUE(D1.empty());
}

TEST(RandomAccessDeque, EmplaceBack) {
  struct Point {
    int X, Y;
    Point(int X, int Y) : X(X), Y(Y) {}
  };

  RandomAccessDeque<Point, 4> D;
  D.emplace_back(1, 2);
  D.emplace_back(3, 4);

  EXPECT_EQ(D[0].X, 1);
  EXPECT_EQ(D[0].Y, 2);
  EXPECT_EQ(D[1].X, 3);
  EXPECT_EQ(D[1].Y, 4);
}

TEST(RandomAccessDeque, ConstAccess) {
  RandomAccessDeque<int, 4> D;
  for (int I = 0; I < 10; ++I)
    D.push_back(I);

  const auto &CD = D;
  EXPECT_EQ(CD[5], 5);
  EXPECT_EQ(CD.front(), 0);
  EXPECT_EQ(CD.back(), 9);

  int Sum = 0;
  for (const int &Val : CD)
    Sum += Val;
  EXPECT_EQ(Sum, 45);
}

TEST(RandomAccessDeque, LargeElements) {
  struct Large {
    char Data[256];
    int Value;
    Large(int V) : Value(V) { llvm::fill(Data, '\0'); }
  };

  RandomAccessDeque<Large, 16> D;
  for (int I = 0; I < 100; ++I)
    D.push_back(Large(I));

  for (uint32_t I = 0; I < D.size(); ++I)
    EXPECT_EQ(D[I].Value, static_cast<int>(I));
}

TEST(RandomAccessDeque, DefaultBucketSize) {
  // Default InitialBucketSize is calculated to fit one page (4KB)
  // For int (4 bytes): 4096/4 = 1024
  RandomAccessDeque<int> D;
  D.push_back(1);
  EXPECT_EQ(D.capacity(), 1024u);
}

TEST(RandomAccessDeque, CustomBucketSize) {
  // Explicit small bucket size
  RandomAccessDeque<int, 4> D;
  D.push_back(1);
  EXPECT_EQ(D.capacity(), 4u);
}

//===----------------------------------------------------------------------===//
// defaultBucketSize tests
//===----------------------------------------------------------------------===//

TEST(DefaultBucketSize, BasicTypes) {
  // int (4 bytes): 4096/4 = 1024
  static_assert(defaultBucketSize<int>() == 1024);
  EXPECT_EQ(defaultBucketSize<int>(), 1024u);

  // char (1 byte): 4096/1 = 4096
  static_assert(defaultBucketSize<char>() == 4096);
  EXPECT_EQ(defaultBucketSize<char>(), 4096u);

  // double (8 bytes): 4096/8 = 512
  static_assert(defaultBucketSize<double>() == 512);
  EXPECT_EQ(defaultBucketSize<double>(), 512u);
}

TEST(DefaultBucketSize, LargeTypes) {
  // 256 bytes: 4096/256 = 16
  struct S256 {
    char data[256];
  };
  static_assert(defaultBucketSize<S256>() == 16);
  EXPECT_EQ(defaultBucketSize<S256>(), 16u);

  // 300 bytes: 4096/300 = 13, bit_floor(13) = 8
  struct S300 {
    char data[300];
  };
  static_assert(defaultBucketSize<S300>() == 8);
  EXPECT_EQ(defaultBucketSize<S300>(), 8u);

  // 1000 bytes: 4096/1000 = 4
  struct S1000 {
    char data[1000];
  };
  static_assert(defaultBucketSize<S1000>() == 4);
  EXPECT_EQ(defaultBucketSize<S1000>(), 4u);

  // 2048 bytes: 4096/2048 = 2
  struct S2048 {
    char data[2048];
  };
  static_assert(defaultBucketSize<S2048>() == 2);
  EXPECT_EQ(defaultBucketSize<S2048>(), 2u);
}

TEST(DefaultBucketSize, VeryLargeTypes) {
  // Larger than a page: should return 1
  struct S5000 {
    char data[5000];
  };
  static_assert(defaultBucketSize<S5000>() == 1);
  EXPECT_EQ(defaultBucketSize<S5000>(), 1u);

  // Exactly one page: 4096/4096 = 1
  struct S4096 {
    char data[4096];
  };
  static_assert(defaultBucketSize<S4096>() == 1);
  EXPECT_EQ(defaultBucketSize<S4096>(), 1u);
}

TEST(DefaultBucketSize, CustomPageSize) {
  // Test with custom page sizes
  // 8KB page, int: 8192/4 = 2048
  static_assert(defaultBucketSize<int, 8192>() == 2048);
  EXPECT_EQ((defaultBucketSize<int, 8192>()), 2048u);

  // 512 byte page, int: 512/4 = 128
  static_assert(defaultBucketSize<int, 512>() == 128);
  EXPECT_EQ((defaultBucketSize<int, 512>()), 128u);
}

} // namespace
} // namespace llvm
