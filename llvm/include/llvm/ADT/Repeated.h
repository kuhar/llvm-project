//===- llvm/ADT/Repeated.h - Repeated value wrapper -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Repeated<T> class, a lightweight wrapper representing
// N copies of the same value. It is designed for use as a function parameter
// (like Twine or ArrayRef) and should not be stored.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_REPEATED_H
#define LLVM_ADT_REPEATED_H

#include <cassert>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace llvm {

/// A single-element wrapper with guaranteed 8-byte alignment, suitable for
/// use as a PointerUnion target in range types. This is an implementation
/// detail of Repeated<T> -- use Repeated<T> directly.
template <typename T>
struct alignas(alignof(void *)) RepeatedStorage {
  T value;
};

/// A random-access iterator that always dereferences to the same value.
template <typename T>
class RepeatedIterator {
public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = ptrdiff_t;
  using pointer = const T *;
  using reference = const T &;

  RepeatedIterator() = default;
  RepeatedIterator(const T *value, ptrdiff_t index)
      : value(value), index(index) {}

  reference operator*() const { return *value; }
  pointer operator->() const { return value; }
  reference operator[](difference_type) const { return *value; }

  RepeatedIterator &operator++() { ++index; return *this; }
  RepeatedIterator operator++(int) { auto tmp = *this; ++index; return tmp; }
  RepeatedIterator &operator--() { --index; return *this; }
  RepeatedIterator operator--(int) { auto tmp = *this; --index; return tmp; }

  RepeatedIterator &operator+=(difference_type n) { index += n; return *this; }
  RepeatedIterator &operator-=(difference_type n) { index -= n; return *this; }

  friend RepeatedIterator operator+(RepeatedIterator it, difference_type n) {
    return {it.value, it.index + n};
  }
  friend RepeatedIterator operator+(difference_type n, RepeatedIterator it) {
    return it + n;
  }
  friend RepeatedIterator operator-(RepeatedIterator it, difference_type n) {
    return {it.value, it.index - n};
  }
  friend difference_type operator-(RepeatedIterator a, RepeatedIterator b) {
    return a.index - b.index;
  }

  friend bool operator==(RepeatedIterator a, RepeatedIterator b) {
    return a.index == b.index;
  }
  friend bool operator!=(RepeatedIterator a, RepeatedIterator b) {
    return a.index != b.index;
  }
  friend bool operator<(RepeatedIterator a, RepeatedIterator b) {
    return a.index < b.index;
  }
  friend bool operator<=(RepeatedIterator a, RepeatedIterator b) {
    return a.index <= b.index;
  }
  friend bool operator>(RepeatedIterator a, RepeatedIterator b) {
    return a.index > b.index;
  }
  friend bool operator>=(RepeatedIterator a, RepeatedIterator b) {
    return a.index >= b.index;
  }

private:
  const T *value = nullptr;
  ptrdiff_t index = 0;
};

/// A lightweight wrapper representing \p count copies of \p value.
///
/// This is designed for transient use as a function parameter -- like Twine or
/// ArrayRef, it should not be stored. Range types (e.g. TypeRange, ValueRange)
/// can accept Repeated<T> via implicit conversion, avoiding the need to
/// materialize an array of identical elements.
///
/// Repeated<T> is also a proper random-access range: begin()/end() return
/// iterators that always dereference to the same stored value.
template <typename T>
struct Repeated {
  RepeatedStorage<T> storage;
  size_t count;

  template <typename U,
            typename = std::enable_if_t<std::is_constructible_v<T, U &&>>>
  Repeated(size_t count, U &&value)
      : storage{std::forward<U>(value)}, count(count) {}

  using iterator = RepeatedIterator<T>;
  using const_iterator = iterator;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = reverse_iterator;
  using value_type = T;
  using size_type = size_t;

  iterator begin() const { return {&storage.value, 0}; }
  iterator end() const { return {&storage.value, static_cast<ptrdiff_t>(count)}; }
  reverse_iterator rbegin() const { return reverse_iterator(end()); }
  reverse_iterator rend() const { return reverse_iterator(begin()); }
  size_t size() const { return count; }
  bool empty() const { return count == 0; }
  const T &operator[](size_t idx) const {
    assert(idx < size() && "index out of bounds");
    return storage.value;
  }
  const T &front() const {
    assert(!empty() && "front() on empty Repeated");
    return storage.value;
  }
  const T &back() const {
    assert(!empty() && "back() on empty Repeated");
    return storage.value;
  }
};

/// Deduction guide: llvm::Repeated(4, someValue) deduces T as std::decay_t
/// of the value type (e.g. string literals decay to const char*).
template <typename U>
Repeated(size_t, U &&) -> Repeated<std::decay_t<U>>;

} // namespace llvm

#endif // LLVM_ADT_REPEATED_H
