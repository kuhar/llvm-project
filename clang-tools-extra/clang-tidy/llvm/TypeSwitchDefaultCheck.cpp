//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeSwitchDefaultCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringSet.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {

void TypeSwitchDefaultCheck::registerMatchers(MatchFinder *Finder) {
  // Match calls to `Default` method on `llvm::TypeSwitch` where the argument
  // is a lambda expression.
  Finder->addMatcher(
      cxxMemberCallExpr(
          callee(cxxMethodDecl(hasName("Default"),
                               ofClass(hasName("::llvm::TypeSwitch")))),
          hasArgument(0, lambdaExpr().bind("lambda")))
          .bind("call"),
      this);
}

/// Returns the single return statement from a lambda body, or nullptr if the
/// lambda body is not a single return statement.
static const ReturnStmt *getSingleReturnStmt(const LambdaExpr *Lambda) {
  const auto *Body = dyn_cast_or_null<CompoundStmt>(Lambda->getBody());
  if (!Body || Body->size() != 1)
    return nullptr;
  return dyn_cast<ReturnStmt>(Body->body_front());
}

/// Returns true if the expression references any parameter of the call
/// operator.
static bool referencesParam(const Expr *E, const CXXMethodDecl *CallOp) {
  if (!E || !CallOp)
    return false;

  E = E->IgnoreParenImpCasts();
  if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (const auto *PVD = dyn_cast<ParmVarDecl>(DRE->getDecl())) {
      // Check if this parameter belongs to the call operator.
      return PVD->getDeclContext() == CallOp;
    }
  }

  // Recursively check all children.
  for (const Stmt *Child : E->children()) {
    if (const auto *ChildExpr = dyn_cast_or_null<Expr>(Child)) {
      if (referencesParam(ChildExpr, CallOp))
        return true;
    }
  }
  return false;
}

/// Checks if the expression is a safe no-argument factory function call like
/// llvm::failure(), llvm::success(), WalkResult::skip(), etc.
static std::string getSafeFactoryCall(const CallExpr *Call) {
  if (!Call || Call->getNumArgs() != 0)
    return "";

  const FunctionDecl *Callee = Call->getDirectCallee();
  if (!Callee)
    return "";

  const std::string QualName = Callee->getQualifiedNameAsString();

  // Allowlist of safe no-argument factory functions.
  static const llvm::StringSet<> SafeFactoryCalls = {
      "llvm::failure",
      "llvm::success",
      "mlir::failure",
      "mlir::success",
      "llvm::WalkResult::skip",
      "llvm::WalkResult::advance",
      "llvm::WalkResult::interrupt",
      "mlir::WalkResult::skip",
      "mlir::WalkResult::advance",
      "mlir::WalkResult::interrupt",
  };

  if (SafeFactoryCalls.contains(QualName))
    return QualName + "()";

  return "";
}

/// Strips copy/move constructors from the expression to get to the underlying
/// value. For example, `nullopt_t(std::nullopt)` becomes `std::nullopt`.
static const Expr *stripCopyMoveConstructors(const Expr *E) {
  while (E) {
    E = E->IgnoreParenImpCasts();
    if (const auto *Construct = dyn_cast<CXXConstructExpr>(E)) {
      // Only strip copy/move constructors (single argument).
      if (Construct->getNumArgs() == 1) {
        E = Construct->getArg(0);
        continue;
      }
    }
    break;
  }
  return E;
}

/// Gets the text representation for a replacement value. Returns empty string
/// if the expression is not a supported type for simplification.
///
/// Only allows expressions that are safe to evaluate eagerly:
/// - nullptr
/// - std::nullopt
/// - Boolean/integer literals
/// - Safe factory functions (failure(), success(), WalkResult::skip(), etc.)
static std::string getReplacementText(const Expr *RetValue, ASTContext &Ctx) {
  RetValue = stripCopyMoveConstructors(RetValue);

  // Handle nullptr.
  if (isa<CXXNullPtrLiteralExpr>(RetValue))
    return "nullptr";

  // Handle boolean literals.
  if (const auto *BoolLit = dyn_cast<CXXBoolLiteralExpr>(RetValue))
    return BoolLit->getValue() ? "true" : "false";

  // Handle integer literals.
  if (const auto *IntLit = dyn_cast<IntegerLiteral>(RetValue)) {
    llvm::SmallString<16> Str;
    IntLit->getValue().toString(Str, 10,
                                IntLit->getType()->isSignedIntegerType());
    return std::string(Str);
  }

  // Handle std::nullopt.
  if (const auto *DRE = dyn_cast<DeclRefExpr>(RetValue)) {
    if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      if (VD->getQualifiedNameAsString() == "std::nullopt")
        return "std::nullopt";
    }
  }

  // Handle safe factory function calls.
  if (const auto *Call = dyn_cast<CallExpr>(RetValue)) {
    std::string FactoryCall = getSafeFactoryCall(Call);
    if (!FactoryCall.empty())
      return FactoryCall;
  }

  // Don't simplify other expressions - they might have side effects or
  // change evaluation semantics.
  return "";
}

void TypeSwitchDefaultCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CXXMemberCallExpr>("call");
  assert(Call);
  const auto *Lambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda");
  assert(Lambda);

  // Check that the lambda has exactly one parameter (the default value).
  const CXXMethodDecl *CallOp = Lambda->getCallOperator();
  if (!CallOp || CallOp->getNumParams() != 1)
    return;

  // Get the single return statement.
  const ReturnStmt *Return = getSingleReturnStmt(Lambda);
  if (!Return)
    return;

  const Expr *RetValue = Return->getRetValue();
  if (!RetValue)
    return;

  // Check that the return value does not reference the lambda parameter.
  if (referencesParam(RetValue, CallOp))
    return;

  // Don't apply fix-its for code in macros.
  if (Call->getBeginLoc().isMacroID())
    return;

  // Get the replacement text for the return value.
  std::string Replacement = getReplacementText(RetValue, *Result.Context);
  if (Replacement.empty())
    return;

  auto Diag = diag(Lambda->getBeginLoc(),
                   "lambda passed to 'Default' can be simplified to '%0'")
              << Replacement;

  // Replace the lambda with just the return value.
  Diag << FixItHint::CreateReplacement(Lambda->getSourceRange(), Replacement);
}

} // namespace clang::tidy::llvm_check
