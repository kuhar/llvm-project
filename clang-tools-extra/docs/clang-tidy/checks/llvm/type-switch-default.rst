.. title:: clang-tidy - llvm-type-switch-default

llvm-type-switch-default
========================

Simplifies ``llvm::TypeSwitch::Default`` calls that take a lambda returning a
constant value that does not depend on the lambda parameter.

The check transforms:

- ``.Default([](auto) { return foo; })`` to ``.Default(foo)``
- ``.Default([](T) { return foo; })`` to ``.Default(foo)``

This simplification is valid when ``foo`` is of the result type ``R`` or one of
the supported implicit conversions: ``nullptr``, ``std::nullopt``, or
``llvm::failure()``.

Example
-------

Before:

.. code-block:: c++

  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<ConstantOp>([](ConstantOp op) { return success(); })
      .Default([](auto) { return llvm::failure(); });

After:

.. code-block:: c++

  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<ConstantOp>([](ConstantOp op) { return success(); })
      .Default(llvm::failure());

Supported Default Values
------------------------

The check supports simplifying lambdas that return:

- ``nullptr`` - for pointer result types
- ``std::nullopt`` - for ``std::optional`` result types
- ``llvm::failure()`` - for ``llvm::FailureOr`` and similar result types
- ``llvm::success()`` - for ``llvm::LogicalResult`` result types
- ``WalkResult::skip()``, ``WalkResult::advance()``, ``WalkResult::interrupt()``
- Boolean literals (``true``, ``false``)
- Integer literals

Limitations
-----------

The check will not simplify lambdas that:

- Use the lambda parameter in the return expression
- Have multiple statements in the body
- Are located inside macros
