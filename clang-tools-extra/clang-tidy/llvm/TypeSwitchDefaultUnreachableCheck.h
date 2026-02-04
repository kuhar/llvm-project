//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_TYPESWITCHDEFAULTUNREACHABLECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_TYPESWITCHDEFAULTUNREACHABLECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::llvm_check {

/// Simplifies llvm::TypeSwitch Default calls that only contain
/// llvm_unreachable by replacing them with DefaultUnreachable.
///
/// Transforms:
///   .Default([](auto) { llvm_unreachable("message"); })
/// To:
///   .DefaultUnreachable("message")
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/llvm/type-switch-default-unreachable.html
class TypeSwitchDefaultUnreachableCheck : public ClangTidyCheck {
public:
  using ClangTidyCheck::ClangTidyCheck;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
};

} // namespace clang::tidy::llvm_check

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_TYPESWITCHDEFAULTUNREACHABLECHECK_H
