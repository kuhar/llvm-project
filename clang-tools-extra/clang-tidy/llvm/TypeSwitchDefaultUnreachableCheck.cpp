//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeSwitchDefaultUnreachableCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {

void TypeSwitchDefaultUnreachableCheck::registerMatchers(MatchFinder *Finder) {
  // Match calls to `llvm::TypeSwitch::Default` with a lambda expression
  // that contains only a call to llvm_unreachable (via the macro which
  // expands to __builtin_unreachable).
  Finder->addMatcher(
      cxxMemberCallExpr(
          argumentCountIs(1),
          callee(memberExpr(member(cxxMethodDecl(
                                hasName("Default"),
                                ofClass(cxxRecordDecl(
                                    hasName("::llvm::TypeSwitch"))))))
                     .bind("member")),
          hasArgument(0, lambdaExpr().bind("lambda")))
          .bind("call"),
      this);
}

void TypeSwitchDefaultUnreachableCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CXXMemberCallExpr>("call");
  assert(Call);
  const auto *Lambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda");
  assert(Lambda);
  const auto *MemExpr = Result.Nodes.getNodeAs<MemberExpr>("member");
  assert(MemExpr);

  // Get the lambda's call operator to examine its body.
  const CXXMethodDecl *CallOp = Lambda->getCallOperator();
  if (!CallOp)
    return;

  const Stmt *Body = CallOp->getBody();
  if (!Body)
    return;

  const auto *CompoundBody = dyn_cast<CompoundStmt>(Body);
  if (!CompoundBody)
    return;

  // The lambda body should contain exactly one statement.
  if (CompoundBody->size() != 1)
    return;

  const Stmt *OnlyStmt = *CompoundBody->body_begin();

  // Handle the case where the statement is wrapped in an ExprWithCleanups.
  if (const auto *Cleanups = dyn_cast<ExprWithCleanups>(OnlyStmt))
    OnlyStmt = Cleanups->getSubExpr();

  // The statement should be a call expression (to llvm_unreachable, which
  // expands to __builtin_unreachable, or directly to llvm_unreachable as
  // a function-like macro call).
  const CallExpr *UnreachableCall = dyn_cast<CallExpr>(OnlyStmt);
  if (!UnreachableCall)
    return;

  // Check if this is __builtin_unreachable (which llvm_unreachable expands to).
  const FunctionDecl *Callee = UnreachableCall->getDirectCallee();
  if (!Callee || Callee->getBuiltinID() != Builtin::BI__builtin_unreachable)
    return;

  // Extract the message from the llvm_unreachable macro.
  // The macro is defined as:
  //   #define llvm_unreachable(msg) __builtin_unreachable()
  // So we need to get the original source text to find the message.
  SourceManager &SM = *Result.SourceManager;
  const LangOptions &LangOpts = Result.Context->getLangOpts();

  // Get the source range of the lambda body content (inside the braces).
  SourceLocation BodyBegin = CompoundBody->getLBracLoc().getLocWithOffset(1);
  SourceLocation BodyEnd = CompoundBody->getRBracLoc();

  // Get the original source text of the lambda body.
  CharSourceRange BodyRange =
      CharSourceRange::getCharRange(BodyBegin, BodyEnd);
  StringRef BodyText = Lexer::getSourceText(BodyRange, SM, LangOpts).trim();

  // Parse the llvm_unreachable call to extract the message.
  // Expected format: llvm_unreachable("message")
  if (!BodyText.starts_with("llvm_unreachable"))
    return;

  // Find the opening and closing parentheses.
  size_t OpenParen = BodyText.find('(');
  size_t CloseParen = BodyText.rfind(')');
  if (OpenParen == StringRef::npos || CloseParen == StringRef::npos ||
      CloseParen <= OpenParen)
    return;

  // Extract the argument (including quotes).
  StringRef MessageArg =
      BodyText.substr(OpenParen + 1, CloseParen - OpenParen - 1).trim();

  // Handle optional semicolon after llvm_unreachable.
  if (BodyText.ends_with(";"))
    BodyText = BodyText.drop_back(1).trim();

  // Build the replacement text.
  std::string Replacement = "DefaultUnreachable(" + MessageArg.str() + ")";

  // Get the range to replace: from ".Default" to the closing parenthesis
  // of the call.
  SourceLocation DotLoc = MemExpr->getOperatorLoc();
  SourceLocation CallEnd = Call->getEndLoc();

  // Skip past the closing parenthesis.
  SourceLocation RParenLoc =
      Lexer::getLocForEndOfToken(CallEnd, 0, SM, LangOpts);

  // Handle macros: don't emit fix-its if parts are in macros.
  if (DotLoc.isMacroID() || CallEnd.isMacroID())
    return;

  auto Diag = diag(Call->getExprLoc(),
                   "use 'DefaultUnreachable' instead of 'Default' with "
                   "'llvm_unreachable'");

  // Create a fix-it to replace ".Default(...)" with ".DefaultUnreachable(msg)".
  SourceRange ReplacementRange(DotLoc.getLocWithOffset(1), CallEnd);
  Diag << FixItHint::CreateReplacement(ReplacementRange, Replacement);
}

} // namespace clang::tidy::llvm_check
