//===- llvm/ADT/PointerUnion.h - Discriminated Union of 2 Ptrs --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the PointerUnion class, which is a discriminated union of
/// pointer types.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_POINTERUNION_H
#define LLVM_ADT_POINTERUNION_H

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/PointerIntPair.h" // For detail::PunnedPointer.
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace llvm {

namespace pointer_union_detail {

/// Determine the number of bits required to store integers with values < n.
/// This is ceil(log2(n)).
constexpr int bitsRequired(unsigned n) {
  return n == 0 ? 0 : llvm::bit_width_constexpr(n - 1);
}

template <typename... Ts> constexpr int lowBitsAvailable() {
  return std::min<int>({PointerLikeTypeTraits<Ts>::NumLowBitsAvailable...});
}

/// True if all types have enough low bits for a fixed-width tag.
template <typename... PTs> constexpr bool isSingleTier() {
  return lowBitsAvailable<PTs...>() >= bitsRequired(sizeof...(PTs));
}

/// True if types are in non-decreasing NumLowBitsAvailable order.
template <typename... PTs> constexpr bool typesInAscendingBitOrder() {
  if constexpr (sizeof...(PTs) <= 1)
    return true;
  else {
    int bits[] = {PointerLikeTypeTraits<PTs>::NumLowBitsAvailable...};
    for (size_t i = 1; i < sizeof...(PTs); ++i)
      if (bits[i] < bits[i - 1])
        return false;
    return true;
  }
}

/// True if the variable-length encoding has enough capacity for all types.
template <typename... PTs> constexpr bool extendedTagsFit() {
  if constexpr (sizeof...(PTs) == 0)
    return true;
  else {
    constexpr size_t N = sizeof...(PTs);
    int bits[] = {PointerLikeTypeTraits<PTs>::NumLowBitsAvailable...};
    int prevBits = 0;
    size_t i = 0;
    while (i < N) {
      int tierBits = bits[i];
      int newBits = tierBits - prevBits;
      size_t tierEnd = i;
      while (tierEnd < N && bits[tierEnd] == tierBits)
        ++tierEnd;
      bool isLastTier = (tierEnd == N);
      size_t typesInTier = tierEnd - i;
      size_t capacity =
          isLastTier ? size_t(1) << newBits : (size_t(1) << newBits) - 1;
      if (typesInTier > capacity)
        return false;
      prevBits = tierBits;
      i = tierEnd;
    }
    return true;
  }
}

/// Tag descriptor for one type in the union.
struct TagEntry {
  intptr_t value; ///< Bit pattern stored in the low bits.
  intptr_t mask;  ///< Mask covering all tag bits for this entry.
};

/// Compute fixed-width tag table (all types have enough bits for the tag).
template <typename... PTs>
constexpr std::array<TagEntry, sizeof...(PTs)> computeFixedTags() {
  constexpr size_t N = sizeof...(PTs);
  constexpr intptr_t mask = (intptr_t(1) << bitsRequired(N)) - 1;
  std::array<TagEntry, N> result = {};
  for (size_t i = 0; i < N; ++i) {
    result[i].value = intptr_t(i);
    result[i].mask = mask;
  }
  return result;
}

/// Compute variable-length tag table for multi-tier.  Types must be in
/// ascending NumLowBitsAvailable order.  Groups types into tiers by bit count;
/// each non-final tier reserves one code as an escape prefix.
template <typename... PTs>
constexpr std::array<TagEntry, sizeof...(PTs)> computeExtendedTags() {
  constexpr size_t N = sizeof...(PTs);
  std::array<TagEntry, N> result = {};
  if constexpr (N == 0)
    return result;
  else {
    int bits[] = {PointerLikeTypeTraits<PTs>::NumLowBitsAvailable...};
    intptr_t escapePrefix = 0;
    int prevBits = 0;
    size_t i = 0;
    while (i < N) {
      int tierBits = bits[i];
      int newBits = tierBits - prevBits;
      size_t tierEnd = i;
      while (tierEnd < N && bits[tierEnd] == tierBits)
        ++tierEnd;
      for (size_t j = 0; j < tierEnd - i; ++j) {
        result[i + j].value = escapePrefix | (intptr_t(j) << prevBits);
        result[i + j].mask = (intptr_t(1) << tierBits) - 1;
      }
      intptr_t escapeCode = (intptr_t(1) << newBits) - 1;
      escapePrefix |= escapeCode << prevBits;
      prevBits = tierBits;
      i = tierEnd;
    }
    return result;
  }
}

/// CRTP base that generates non-template constructors and assignment operators
/// for each type in the union.  Non-template constructors allow implicit
/// conversions (derived-to-base, non-const-to-const) matching the historical
/// PointerUnion behavior.
template <typename Derived, int I, typename... Types>
class PointerUnionMembers;

template <typename Derived, int I>
class PointerUnionMembers<Derived, I> {
protected:
  detail::PunnedPointer<void *> Val;
  PointerUnionMembers() : Val(intptr_t(0)) {}

  template <typename To, typename From, typename Enable>
  friend struct ::llvm::CastInfo;
  template <typename> friend struct ::llvm::PointerLikeTypeTraits;
};

template <typename Derived, int I, typename Type, typename... Types>
class PointerUnionMembers<Derived, I, Type, Types...>
    : public PointerUnionMembers<Derived, I + 1, Types...> {
  using Base = PointerUnionMembers<Derived, I + 1, Types...>;

public:
  using Base::Base;
  PointerUnionMembers() = default;

  PointerUnionMembers(Type V) { this->Val = Derived::encode(V); }

  using Base::operator=;
  Derived &operator=(Type V) {
    this->Val = Derived::encode(V);
    return static_cast<Derived &>(*this);
  }
};

} // end namespace pointer_union_detail

/// A discriminated union of two or more pointer types, with the discriminator
/// in the low bits of the pointer.
///
/// This implementation is extremely efficient in space due to leveraging the
/// low bits of the pointer, while exposing a natural and type-safe API.
///
/// When all types have enough alignment for a fixed-width tag (single-tier),
/// the tag is placed in the high end of the available low bits, leaving spare
/// low bits for nesting in PointerIntPair or SmallPtrSet.  When types have
/// heterogeneous alignment (multi-tier), a variable-length escape-encoded tag
/// is used; in that case, types must be listed in non-decreasing
/// NumLowBitsAvailable order.
///
/// Common use patterns would be something like this:
///    PointerUnion<int*, float*> P;
///    P = (int*)0;
///    printf("%d %d", P.is<int*>(), P.is<float*>());  // prints "1 0"
///    X = P.get<int*>();     // ok.
///    Y = P.get<float*>();   // runtime assertion failure.
///    Z = P.get<double*>();  // compile time failure.
///    P = (float*)0;
///    Y = P.get<float*>();   // ok.
///    X = P.get<int*>();     // runtime assertion failure.
///    PointerUnion<int*, int*> Q; // compile time failure.
template <typename... PTs>
class PointerUnion
    : public pointer_union_detail::PointerUnionMembers<PointerUnion<PTs...>, 0,
                                                       PTs...> {
  static_assert(TypesAreDistinct<PTs...>::value,
                "PointerUnion alternative types cannot be repeated");

  using Base = typename PointerUnion::PointerUnionMembers;
  using First = TypeAtIndex<0, PTs...>;

  template <typename, int, typename...>
  friend class pointer_union_detail::PointerUnionMembers;
  template <typename To, typename From, typename Enable>
  friend struct CastInfo;
  template <typename> friend struct PointerLikeTypeTraits;

  // --- Lazy tag configuration ---
  // These are constexpr *functions*, not static data members, so their bodies
  // are only instantiated when called.  This avoids evaluating alignof() on
  // potentially incomplete types at class-definition time.

  static constexpr bool singleTier() {
    return pointer_union_detail::isSingleTier<PTs...>();
  }

  static constexpr int minBits() {
    return pointer_union_detail::lowBitsAvailable<PTs...>();
  }

  static constexpr int tagBits() {
    return pointer_union_detail::bitsRequired(sizeof...(PTs));
  }

  /// In single-tier mode, the tag is shifted to the high end of the available
  /// low bits so that the lowest bits remain free for nesting.  In multi-tier
  /// mode, the tag starts at bit 0.
  static constexpr int tagShift() {
    return singleTier() ? (minBits() - tagBits()) : 0;
  }

  static constexpr auto tagTable() {
    if constexpr (sizeof...(PTs) == 0) {
      return std::array<pointer_union_detail::TagEntry, 0>{};
    } else if constexpr (singleTier()) {
      return pointer_union_detail::computeFixedTags<PTs...>();
    } else {
      static_assert(pointer_union_detail::typesInAscendingBitOrder<PTs...>(),
                    "Multi-tier PointerUnion types must be in ascending "
                    "NumLowBitsAvailable order");
      static_assert(pointer_union_detail::extendedTagsFit<PTs...>(),
                    "Too many types for the available low bits");
      return pointer_union_detail::computeExtendedTags<PTs...>();
    }
  }

  // Multi-tier isNull: check membership in the sparse set of tag values.
  template <size_t... Is>
  static constexpr bool isNullCheck(intptr_t v, std::index_sequence<Is...>) {
    constexpr auto table = tagTable();
    return ((v == table[Is].value) || ...);
  }

  template <typename T>
  static intptr_t encode(T V) {
    constexpr auto table = tagTable();
    constexpr int shift = tagShift();
    constexpr size_t Idx = FirstIndexOfType<T, PTs...>::value;
    static_assert(table[0].value == 0,
                  "First type must have tag value 0 for getAddrOfPtr1");
    void *VoidPtr =
        const_cast<void *>(PointerLikeTypeTraits<T>::getAsVoidPointer(V));
    intptr_t ptrInt = reinterpret_cast<intptr_t>(VoidPtr);
    assert((ptrInt & (table[Idx].mask << shift)) == 0 &&
           "Pointer low bits collide with tag");
    return ptrInt | (table[Idx].value << shift);
  }

public:

  PointerUnion() = default;
  PointerUnion(std::nullptr_t) : PointerUnion() {}
  using Base::Base;
  using Base::operator=;

  /// Assignment from nullptr clears the union, resetting to the first type.
  const PointerUnion &operator=(std::nullptr_t) {
    this->Val = intptr_t(0);
    return *this;
  }

  /// Test if the pointer held in the union is null, regardless of
  /// which type it is.
  bool isNull() const {
    if constexpr (singleTier()) {
      // All null values fit entirely within the tag field.
      return static_cast<uintptr_t>(this->Val.asInt()) <
             (uintptr_t(1) << minBits());
    } else {
      return isNullCheck(this->Val.asInt(), std::index_sequence_for<PTs...>{});
    }
  }

  explicit operator bool() const { return !isNull(); }

  template <typename T> [[deprecated("Use isa instead")]] bool is() const {
    return isa<T>(*this);
  }

  template <typename T> [[deprecated("Use cast instead")]] T get() const {
    assert(isa<T>(*this) && "Invalid accessor called");
    return cast<T>(*this);
  }

  template <typename T> inline T dyn_cast() const {
    return llvm::dyn_cast_if_present<T>(*this);
  }

  /// If the union is set to the first pointer type get an address pointing to
  /// it.
  First const *getAddrOfPtr1() const {
    return const_cast<PointerUnion *>(this)->getAddrOfPtr1();
  }

  /// If the union is set to the first pointer type get an address pointing to
  /// it.
  First *getAddrOfPtr1() {
    assert(isa<First>(*this) && "Val is not the first pointer");
    // tag == 0 for first type, so asInt() is the raw pointer value.
    assert(
        PointerLikeTypeTraits<First>::getAsVoidPointer(cast<First>(*this)) ==
            reinterpret_cast<void *>(this->Val.asInt()) &&
        "Can't get the address because PointerLikeTypeTraits changes the ptr");
    return const_cast<First *>(
        reinterpret_cast<const First *>(this->Val.getPointerAddress()));
  }

  void *getOpaqueValue() const {
    return reinterpret_cast<void *>(this->Val.asInt());
  }

  static inline PointerUnion getFromOpaqueValue(void *VP) {
    PointerUnion V;
    V.Val = reinterpret_cast<intptr_t>(VP);
    return V;
  }

  friend bool operator==(PointerUnion lhs, PointerUnion rhs) {
    return lhs.getOpaqueValue() == rhs.getOpaqueValue();
  }

  friend bool operator!=(PointerUnion lhs, PointerUnion rhs) {
    return lhs.getOpaqueValue() != rhs.getOpaqueValue();
  }

  friend bool operator<(PointerUnion lhs, PointerUnion rhs) {
    return lhs.getOpaqueValue() < rhs.getOpaqueValue();
  }
};

// Specialization of CastInfo for PointerUnion.
template <typename To, typename... PTs>
struct CastInfo<To, PointerUnion<PTs...>>
    : public DefaultDoCastIfPossible<To, PointerUnion<PTs...>,
                                     CastInfo<To, PointerUnion<PTs...>>> {
  using From = PointerUnion<PTs...>;

  static inline bool isPossible(From &f) {
    constexpr auto table = From::tagTable();
    constexpr int shift = From::tagShift();
    constexpr size_t Idx = FirstIndexOfType<To, PTs...>::value;
    intptr_t v = reinterpret_cast<intptr_t>(f.getOpaqueValue());
    constexpr intptr_t mask = table[Idx].mask << shift;
    constexpr intptr_t value = table[Idx].value << shift;
    return (v & mask) == value;
  }

  static To doCast(From &f) {
    assert(isPossible(f) && "cast to an incompatible type!");
    constexpr intptr_t ptrMask =
        ~((intptr_t(1) << PointerLikeTypeTraits<To>::NumLowBitsAvailable) - 1);
    void *ptr = reinterpret_cast<void *>(
        reinterpret_cast<intptr_t>(f.getOpaqueValue()) & ptrMask);
    return PointerLikeTypeTraits<To>::getFromVoidPointer(ptr);
  }

  static inline To castFailed() { return To(); }
};

template <typename To, typename... PTs>
struct CastInfo<To, const PointerUnion<PTs...>>
    : public ConstStrippingForwardingCast<
          To, const PointerUnion<PTs...>,
          CastInfo<To, PointerUnion<PTs...>>> {};

// Teach SmallPtrSet that PointerUnion is "basically a pointer".
// In single-tier mode, spare low bits are available for nesting.
// This specialization is only instantiated when used (lazy), so
// PointerLikeTypeTraits<PTs> / alignof() are not evaluated for
// incomplete types.
template <typename... PTs>
struct PointerLikeTypeTraits<PointerUnion<PTs...>> {
  using Union = PointerUnion<PTs...>;

  static inline void *getAsVoidPointer(const Union &P) {
    return P.getOpaqueValue();
  }

  static inline Union getFromVoidPointer(void *P) {
    return Union::getFromOpaqueValue(P);
  }

  static constexpr int NumLowBitsAvailable = Union::tagShift();
};

// Teach DenseMap how to use PointerUnions as keys.
template <typename... PTs> struct DenseMapInfo<PointerUnion<PTs...>> {
  using Union = PointerUnion<PTs...>;
  using FirstInfo = DenseMapInfo<TypeAtIndex<0, PTs...>>;

  static inline Union getEmptyKey() { return Union(FirstInfo::getEmptyKey()); }

  static inline Union getTombstoneKey() {
    return Union(FirstInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const Union &UnionVal) {
    intptr_t key = (intptr_t)UnionVal.getOpaqueValue();
    return DenseMapInfo<intptr_t>::getHashValue(key);
  }

  static bool isEqual(const Union &LHS, const Union &RHS) {
    return LHS == RHS;
  }
};

} // end namespace llvm

#endif // LLVM_ADT_POINTERUNION_H
