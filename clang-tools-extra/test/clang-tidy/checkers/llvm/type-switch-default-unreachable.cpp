// RUN: %check_clang_tidy -std=c++17-or-later %s llvm-type-switch-default-unreachable %t

// Mock llvm_unreachable.
#define llvm_unreachable(msg) __builtin_unreachable()

namespace llvm {

template <typename T, typename ResultT = void>
class TypeSwitch {
public:
  TypeSwitch(T) {}

  template <typename CaseT, typename CallableT>
  TypeSwitch &Case(CallableT &&) { return *this; }

  template <typename CallableT>
  TypeSwitch &Case(CallableT &&) { return *this; }

  template <typename CallableT>
  ResultT Default(CallableT &&) { return ResultT(); }

  ResultT DefaultUnreachable(const char * = "Fell off the end of a type-switch") {
    return ResultT();
  }
};

// Specialization for void return type.
template <typename T>
class TypeSwitch<T, void> {
public:
  TypeSwitch(T) {}

  template <typename CaseT, typename CallableT>
  TypeSwitch &Case(CallableT &&) { return *this; }

  template <typename CallableT>
  TypeSwitch &Case(CallableT &&) { return *this; }

  template <typename CallableT>
  void Default(CallableT &&) {}

  void DefaultUnreachable(const char * = "Fell off the end of a type-switch") {}
};

} // namespace llvm

// Test types.
struct Base {};
struct DerivedA : Base {};
struct DerivedB : Base {};

//===----------------------------------------------------------------------===//
// Positive test cases - should trigger warnings with fix-its.
//===----------------------------------------------------------------------===//

void test_default_with_unreachable_auto(Base *base) {
  llvm::TypeSwitch<Base *>(base)
      .Case<DerivedA>([](DerivedA *) {})
      .Default([](auto) { llvm_unreachable("unhandled type"); });
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use 'DefaultUnreachable' instead of 'Default' with 'llvm_unreachable'
  // CHECK-FIXES: .DefaultUnreachable("unhandled type");
}

void test_default_with_unreachable_auto_ref(Base *base) {
  llvm::TypeSwitch<Base *>(base)
      .Case<DerivedA>([](DerivedA *) {})
      .Default([&](auto) { llvm_unreachable("unhandled case"); });
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use 'DefaultUnreachable' instead of 'Default' with 'llvm_unreachable'
  // CHECK-FIXES: .DefaultUnreachable("unhandled case");
}

void test_default_with_unreachable_explicit_type(Base *base) {
  llvm::TypeSwitch<Base *>(base)
      .Case<DerivedA>([](DerivedA *) {})
      .Default([](Base *) { llvm_unreachable("unexpected type"); });
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use 'DefaultUnreachable' instead of 'Default' with 'llvm_unreachable'
  // CHECK-FIXES: .DefaultUnreachable("unexpected type");
}

void test_default_with_unreachable_no_param_name(Base *base) {
  llvm::TypeSwitch<Base *>(base)
      .Case<DerivedA>([](DerivedA *) {})
      .Default([](auto) { llvm_unreachable("no param name"); });
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use 'DefaultUnreachable' instead of 'Default' with 'llvm_unreachable'
  // CHECK-FIXES: .DefaultUnreachable("no param name");
}

void test_default_with_unreachable_capture_this() {
  struct S {
    void method(Base *base) {
      llvm::TypeSwitch<Base *>(base)
          .Case<DerivedA>([](DerivedA *) {})
          .Default([this](auto) { llvm_unreachable("captured this"); });
      // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use 'DefaultUnreachable' instead of 'Default' with 'llvm_unreachable'
      // CHECK-FIXES: .DefaultUnreachable("captured this");
    }
  };
}

void test_default_with_unreachable_multiline(Base *base) {
  llvm::TypeSwitch<Base *>(base)
      .Case<DerivedA>([](DerivedA *) {})
      .Default([](auto) {
        llvm_unreachable("multiline");
      });
  // CHECK-MESSAGES: :[[@LINE-3]]:8: warning: use 'DefaultUnreachable' instead of 'Default' with 'llvm_unreachable'
  // CHECK-FIXES: .DefaultUnreachable("multiline");
}

void test_with_result_type(Base *base) {
  llvm::TypeSwitch<Base *, int>(base)
      .Case<DerivedA>([](DerivedA *) { return 1; })
      .Default([](auto) -> int { llvm_unreachable("with result"); });
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use 'DefaultUnreachable' instead of 'Default' with 'llvm_unreachable'
  // CHECK-FIXES: .DefaultUnreachable("with result");
}

void test_fully_qualified(Base *base) {
  ::llvm::TypeSwitch<Base *>(base)
      .Case<DerivedA>([](DerivedA *) {})
      .Default([](auto) { llvm_unreachable("fully qualified"); });
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use 'DefaultUnreachable' instead of 'Default' with 'llvm_unreachable'
  // CHECK-FIXES: .DefaultUnreachable("fully qualified");
}

namespace llvm {
void test_inside_llvm_namespace(Base *base) {
  TypeSwitch<Base *>(base)
      .Case<DerivedA>([](DerivedA *) {})
      .Default([](auto) { llvm_unreachable("inside namespace"); });
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use 'DefaultUnreachable' instead of 'Default' with 'llvm_unreachable'
  // CHECK-FIXES: .DefaultUnreachable("inside namespace");
}
} // namespace llvm

//===----------------------------------------------------------------------===//
// Negative test cases - should NOT trigger any warnings.
//===----------------------------------------------------------------------===//

void test_negative_default_with_other_code(Base *base) {
  // Default lambda does more than just llvm_unreachable.
  llvm::TypeSwitch<Base *>(base)
      .Case<DerivedA>([](DerivedA *) {})
      .Default([](auto x) {
        (void)x;
        llvm_unreachable("has other code");
      });
}

void test_negative_default_with_return_before_unreachable(Base *base) {
  // Default lambda has a return statement.
  llvm::TypeSwitch<Base *, int>(base)
      .Case<DerivedA>([](DerivedA *) { return 1; })
      .Default([](auto) {
        return 0;
        llvm_unreachable("unreachable after return");
      });
}

void test_negative_default_without_unreachable(Base *base) {
  // Default lambda does not use llvm_unreachable.
  llvm::TypeSwitch<Base *>(base)
      .Case<DerivedA>([](DerivedA *) {})
      .Default([](auto) {});
}

void test_negative_default_with_value(Base *base) {
  // Default with a value, not a lambda.
  llvm::TypeSwitch<Base *, int>(base)
      .Case<DerivedA>([](DerivedA *) { return 1; })
      .Default(0);
}

void test_negative_already_using_default_unreachable(Base *base) {
  // Already using DefaultUnreachable.
  llvm::TypeSwitch<Base *>(base)
      .Case<DerivedA>([](DerivedA *) {})
      .DefaultUnreachable("already correct");
}

// Non-TypeSwitch class with Default method.
struct OtherClass {
  template <typename F>
  OtherClass &Default(F &&) { return *this; }
};

void test_negative_non_type_switch() {
  OtherClass().Default([](auto) { llvm_unreachable("not TypeSwitch"); });
}

// TypeSwitch in non-llvm namespace.
namespace other {
template <typename T, typename R = void>
struct TypeSwitch {
  TypeSwitch(T) {}
  template <typename F>
  void Default(F &&) {}
};
} // namespace other

void test_negative_non_llvm_namespace(Base *base) {
  other::TypeSwitch<Base *>(base)
      .Default([](auto) { llvm_unreachable("not llvm::TypeSwitch"); });
}
