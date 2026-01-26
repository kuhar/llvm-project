//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_TYPESWITCHDEFAULTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_TYPESWITCHDEFAULTCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::llvm_check {

/// Simplifies `TypeSwitch::Default` calls that take a lambda returning a
/// constant value.
///
/// Rewrites:
///   `.Default([](auto) { return foo; })` --> `.Default(foo)`
///   `.Default([](T) { return foo; })`    --> `.Default(foo)`
///
/// When `foo` is of the result type or a supported implicit conversion
/// (`nullptr`, `std::nullopt`, `llvm::failure()`).
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/llvm/type-switch-default.html
class TypeSwitchDefaultCheck : public ClangTidyCheck {
public:
  TypeSwitchDefaultCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
};

} // namespace clang::tidy::llvm_check

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_TYPESWITCHDEFAULTCHECK_H
