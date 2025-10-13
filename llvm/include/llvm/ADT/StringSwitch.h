//===--- StringSwitch.h - Switch-on-literal-string Construct --------------===/
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===/
///
/// \file
///  This file implements the StringSwitch template, which mimics a switch()
///  statement whose cases are string literals.
///
//===----------------------------------------------------------------------===/
#ifndef LLVM_ADT_STRINGSWITCH_H
#define LLVM_ADT_STRINGSWITCH_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstring>
#include <optional>
#include <tuple>
#include <utility>

namespace llvm {

/// A switch()-like statement whose cases are string literals.
///
/// The StringSwitch class is a simple form of a switch() statement that
/// determines whether the given string matches one of the given string
/// literals. The template type parameter \p T is the type of the value that
/// will be returned from the string-switch expression. For example,
/// the following code switches on the name of a color in \c argv[i]:
///
/// \code
/// Color color = StringSwitch<Color>(argv[i])
///   .Case("red", Red)
///   .Case("orange", Orange)
///   .Case("yellow", Yellow)
///   .Case("green", Green)
///   .Case("blue", Blue)
///   .Case("indigo", Indigo)
///   .Cases("violet", "purple", Violet)
///   .Default(UnknownColor);
/// \endcode
template<typename T, typename R = T>
class StringSwitch {
  /// The string we are matching.
  const StringRef Str;

  /// The pointer to the result of this switch statement, once known,
  /// null before that.
  std::optional<T> Result;

public:
  explicit StringSwitch(StringRef S)
  : Str(S), Result() { }

  // StringSwitch is not copyable.
  StringSwitch(const StringSwitch &) = delete;

  // StringSwitch is not assignable due to 'Str' being 'const'.
  void operator=(const StringSwitch &) = delete;
  void operator=(StringSwitch &&other) = delete;

  StringSwitch(StringSwitch &&other)
    : Str(other.Str), Result(std::move(other.Result)) { }

  ~StringSwitch() = default;

  // Case-sensitive case matchers
  StringSwitch &Case(StringLiteral S, T Value) {
    CaseImpl(Value, S);
    return *this;
  }

  StringSwitch& EndsWith(StringLiteral S, T Value) {
    if (!Result && Str.ends_with(S)) {
      Result = std::move(Value);
    }
    return *this;
  }

  StringSwitch& StartsWith(StringLiteral S, T Value) {
    if (!Result && Str.starts_with(S)) {
      Result = std::move(Value);
    }
    return *this;
  }

  // The last argument is the Value to return when any of the cases match.
  template <typename... Ts>
  StringSwitch &Cases(StringLiteral First, StringLiteral Second, Ts &&...Rest) {
    using LastType = TypeAtIndex<sizeof...(Ts) - 1, Ts...>;
    T Value = std::forward<LastType&&>((..., Rest));
    return CasesDispatch</*Lower=*/false>(
        Value, std::forward_as_tuple(First, Second, std::forward<Ts>(Rest)...),
        std::make_index_sequence<sizeof...(Rest) + 1>{});
  }

  // Case-insensitive case matchers.
  StringSwitch &CaseLower(StringLiteral S, T Value) {
    CaseLowerImpl(Value, S);
    return *this;
  }

  StringSwitch &EndsWithLower(StringLiteral S, T Value) {
    if (!Result && Str.ends_with_insensitive(S))
      Result = Value;

    return *this;
  }

  StringSwitch &StartsWithLower(StringLiteral S, T Value) {
    if (!Result && Str.starts_with_insensitive(S))
      Result = std::move(Value);

    return *this;
  }

  // Returns the `Value` when any of the (string literal) Cases match
  // (case-insensitive). The last argument is the `Value` to return.
  template <typename... Ts>
  StringSwitch &CasesLower(StringLiteral First, StringLiteral Second,
                           Ts &&...Rest) {
    using LastType = TypeAtIndex<sizeof...(Ts) - 1, Ts...>;
    T Value = std::forward<LastType&&>((..., Rest));
    return CasesDispatch</*Lower=*/true>(
        Value, std::forward_as_tuple(First, Second, std::forward<Ts>(Rest)...),
        std::make_index_sequence<sizeof...(Rest) + 1>{});
  }

  [[nodiscard]] R Default(T Value) {
    if (Result)
      return std::move(*Result);
    return Value;
  }

  /// Declare default as unreachable, making sure that all cases were handled.
  [[nodiscard]] R DefaultUnreachable(
      const char *Message = "Fell off the end of a string-switch") {
    if (Result)
      return std::move(*Result);
    llvm_unreachable(Message);
  }

  [[nodiscard]] operator R() { return DefaultUnreachable(); }

private:
  // Returns true when `Str` matches the `S` argument, and stores the result.
  bool CaseImpl(T &Value, StringLiteral S) {
    if (!Result && Str == S) {
      Result = std::move(Value);
      return true;
    }
    return false;
  }

  // Returns true when `Str` matches the `S` argument (case-insensitive), and
  // stores the result.
  bool CaseLowerImpl(T &Value, StringLiteral S) {
    if (!Result && Str.equals_insensitive(S)) {
      Result = std::move(Value);
      return true;
    }
    return false;
  }

  // Since we cannot force conversion to StringLiterals with variadic templates,
  // provide a helper function to handle char arrays. We can't just rely on the
  // StringLiteral constructor because of the enable_if attribute that won't
  // work because the string is not a constant expression here.
  template <size_t N> static StringLiteral ToStringLiteral(const char (&S)[N]) {
    // StringLiteral's constructor enforces the same invariant on some
    // toolchains.
    assert(strlen(S) == N - 1 && "Invalid string literal");
    return StringLiteral::withInnerNUL(S);
  }
  static StringLiteral ToStringLiteral(StringLiteral S) { return S; }

  // Implements matching over multiple cases, given `Refs` with all the string
  // literals in the front and the `Value` in the back. The index sequence
  // contains one fewer index than the tuple size so that we can extract all
  // case strings and skip the `Value`.
  template <bool Lower, typename... Ts, size_t... Idxs>
  StringSwitch &CasesDispatch(T &Value, std::tuple<Ts...> &&Refs,
                              std::index_sequence<Idxs...>) {
    if constexpr (Lower) {
      (... || CaseLowerImpl(Value, ToStringLiteral(std::get<Idxs>(Refs))));
    } else {
      (... || CaseImpl(Value, ToStringLiteral(std::get<Idxs>(Refs))));
    }
    return *this;
  }
};

} // end namespace llvm

#endif // LLVM_ADT_STRINGSWITCH_H
