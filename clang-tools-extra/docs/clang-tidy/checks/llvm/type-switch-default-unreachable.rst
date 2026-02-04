.. title:: clang-tidy - llvm-type-switch-default-unreachable

llvm-type-switch-default-unreachable
====================================

Finds ``llvm::TypeSwitch::Default`` calls with a lambda that only contains
``llvm_unreachable`` and suggests using ``DefaultUnreachable`` instead.

The ``DefaultUnreachable`` method is a cleaner and more direct way to express
that all cases should be handled by the ``Case`` methods and reaching the
default is a programming error.

Example
-------

.. code-block:: c++

  llvm::TypeSwitch<Type>(type)
      .Case<Float16Type>([](Float16Type t) { return "f16"; })
      .Case<Float32Type>([](Float32Type t) { return "f32"; })
      .Default([](auto) { llvm_unreachable("unhandled type"); });

Transforms to:

.. code-block:: c++

  llvm::TypeSwitch<Type>(type)
      .Case<Float16Type>([](Float16Type t) { return "f16"; })
      .Case<Float32Type>([](Float32Type t) { return "f32"; })
      .DefaultUnreachable("unhandled type");
