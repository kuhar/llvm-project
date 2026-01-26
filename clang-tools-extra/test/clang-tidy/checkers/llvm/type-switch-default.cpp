// RUN: %check_clang_tidy -std=c++17-or-later %s llvm-type-switch-default %t

namespace std {
using nullptr_t = decltype(nullptr);
struct nullopt_t {};
inline constexpr nullopt_t nullopt{};

template <typename T>
class optional {
public:
  optional() = default;
  optional(nullopt_t) {}
  optional(T) {}
};
} // namespace std

namespace llvm {

struct LogicalResult {};
inline LogicalResult failure() { return {}; }
inline LogicalResult success() { return {}; }

template <typename T>
struct FailureOr {
  FailureOr() = default;
  FailureOr(LogicalResult) {}
  FailureOr(T) {}
};

template <typename T, typename ResultT = void>
class TypeSwitch {
  T value;

public:
  TypeSwitch(T v) : value(v) {}

  template <typename CaseT, typename CallableT>
  TypeSwitch &Case(CallableT &&caseFn) {
    return *this;
  }

  template <typename CallableT>
  ResultT Default(CallableT &&defaultFn) {
    return defaultFn(value);
  }

  ResultT Default(ResultT defaultResult) { return defaultResult; }

  template <typename ArgT = ResultT>
  ResultT Default(std::nullptr_t) {
    return ResultT(nullptr);
  }

  template <typename ArgT = ResultT>
  ResultT Default(std::nullopt_t) {
    return ResultT(std::nullopt);
  }

  template <typename ArgT = ResultT>
  ResultT Default(LogicalResult result) {
    return ResultT(result);
  }
};

// Specialization for void.
template <typename T>
class TypeSwitch<T, void> {
  T value;

public:
  TypeSwitch(T v) : value(v) {}

  template <typename CaseT, typename CallableT>
  TypeSwitch &Case(CallableT &&caseFn) {
    return *this;
  }

  template <typename CallableT>
  void Default(CallableT &&defaultFn) {
    defaultFn(value);
  }
};

} // namespace llvm

class Base {
public:
  virtual ~Base() = default;
};
class Derived : public Base {};

//===----------------------------------------------------------------------===//
// Positive test cases: should trigger the check
//===----------------------------------------------------------------------===//

// Test: lambda returning nullptr with auto parameter.
Base *test_nullptr_auto(Base *b) {
  return llvm::TypeSwitch<Base *, Base *>(b)
      .Case<Derived>([](Derived *d) { return d; })
      .Default([](auto) { return nullptr; });
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: lambda passed to 'Default' can be simplified to 'nullptr'
  // CHECK-FIXES: .Default(nullptr);
}

// Test: lambda returning nullptr with explicit type parameter.
Base *test_nullptr_explicit_type(Base *b) {
  return llvm::TypeSwitch<Base *, Base *>(b)
      .Case<Derived>([](Derived *d) { return d; })
      .Default([](Base *) { return nullptr; });
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: lambda passed to 'Default' can be simplified to 'nullptr'
  // CHECK-FIXES: .Default(nullptr);
}

// Test: lambda returning std::nullopt with auto parameter.
std::optional<int> test_nullopt_auto(Base *b) {
  return llvm::TypeSwitch<Base *, std::optional<int>>(b)
      .Case<Derived>([](Derived *) { return 42; })
      .Default([](auto) { return std::nullopt; });
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: lambda passed to 'Default' can be simplified to 'std::nullopt'
  // CHECK-FIXES: .Default(std::nullopt);
}

// Test: lambda returning std::nullopt with explicit type parameter.
std::optional<int> test_nullopt_explicit_type(Base *b) {
  return llvm::TypeSwitch<Base *, std::optional<int>>(b)
      .Case<Derived>([](Derived *) { return 42; })
      .Default([](Base *) { return std::nullopt; });
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: lambda passed to 'Default' can be simplified to 'std::nullopt'
  // CHECK-FIXES: .Default(std::nullopt);
}

// Test: lambda returning llvm::failure() with auto parameter.
llvm::FailureOr<int> test_failure_auto(Base *b) {
  return llvm::TypeSwitch<Base *, llvm::FailureOr<int>>(b)
      .Case<Derived>([](Derived *) { return 42; })
      .Default([](auto) { return llvm::failure(); });
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: lambda passed to 'Default' can be simplified to 'llvm::failure()'
  // CHECK-FIXES: .Default(llvm::failure());
}

// Test: lambda returning llvm::failure() with explicit type parameter.
llvm::FailureOr<int> test_failure_explicit_type(Base *b) {
  return llvm::TypeSwitch<Base *, llvm::FailureOr<int>>(b)
      .Case<Derived>([](Derived *) { return 42; })
      .Default([](Base *) { return llvm::failure(); });
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: lambda passed to 'Default' can be simplified to 'llvm::failure()'
  // CHECK-FIXES: .Default(llvm::failure());
}

// Test: lambda returning an integer literal.
int test_integer_literal(Base *b) {
  return llvm::TypeSwitch<Base *, int>(b)
      .Case<Derived>([](Derived *) { return 42; })
      .Default([](auto) { return 0; });
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: lambda passed to 'Default' can be simplified to '0'
  // CHECK-FIXES: .Default(0);
}

//===----------------------------------------------------------------------===//
// Negative test cases: should NOT trigger the check
//===----------------------------------------------------------------------===//

// Test: lambda uses its parameter in the return value.
Base *test_negative_uses_param(Base *b) {
  return llvm::TypeSwitch<Base *, Base *>(b)
      .Case<Derived>([](Derived *d) { return d; })
      .Default([](auto x) { return x; });
}

// Test: lambda has multiple statements.
Base *test_negative_multiple_statements(Base *b) {
  return llvm::TypeSwitch<Base *, Base *>(b)
      .Case<Derived>([](Derived *d) { return d; })
      .Default([](auto x) {
        (void)x;
        return nullptr;
      });
}

// Test: void-returning TypeSwitch with a lambda that does something.
void test_negative_void_with_side_effect(Base *b) {
  int counter = 0;
  llvm::TypeSwitch<Base *, void>(b)
      .Case<Derived>([&](Derived *) { ++counter; })
      .Default([&](auto) { ++counter; });
}

// Test: lambda returning a variable (not supported - may have side effects).
Base *test_negative_variable(Base *b, Base *defaultValue) {
  return llvm::TypeSwitch<Base *, Base *>(b)
      .Case<Derived>([](Derived *d) { return d; })
      .Default([=](auto) { return defaultValue; });
}

// Test: already simplified - Default takes a value directly.
Base *test_negative_already_simplified(Base *b) {
  return llvm::TypeSwitch<Base *, Base *>(b)
      .Case<Derived>([](Derived *d) { return d; })
      .Default(nullptr);
}

