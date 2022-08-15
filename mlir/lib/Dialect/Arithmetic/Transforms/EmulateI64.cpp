//===- UnsignedWhenEquivalent.cpp - Pass to replace signed operations with
// unsigned
// ones when all their arguments and results are statically non-negative --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cassert>

namespace mlir::arith {
namespace {
class I64EmulationConverter : public TypeConverter {
public:
  explicit I64EmulationConverter(unsigned maxIntegerWidthSupported)
      : maxIntWidth(maxIntegerWidthSupported) {
    // Scalar case.
    addConversion([widestInt = maxIntegerWidthSupported](
                      IntegerType ty) -> Optional<Type> {
      if (ty.getWidth() == 2 * widestInt)
        return VectorType::get({2},
                               IntegerType::get(ty.getContext(), widestInt));
      return None;
    });

    // Vector case.
    addConversion([widestInt = maxIntegerWidthSupported](
                      VectorType ty) -> Optional<Type> {
      if (auto intTy = ty.getElementType().dyn_cast<IntegerType>()) {
        if (intTy.getWidth() == 2 * widestInt) {
          SmallVector<int64_t> newShape = {2};
          llvm::append_range(newShape, ty.getShape());
          return VectorType::get(newShape,
                                 IntegerType::get(ty.getContext(), widestInt));
        }
      }
      return None;
    });
  }

  unsigned getMaxIntegerWidth() const { return maxIntWidth; }

  bool isLegalType(Type ty) const {
    if (auto intTy = ty.dyn_cast<IntegerType>())
      return intTy.getWidth() <= getMaxIntegerWidth();

    if (auto vecTy = ty.dyn_cast<VectorType>())
      return isLegalType(vecTy.getElementType());

    if (auto funcTy = ty.dyn_cast<FunctionType>()) {
      if (!isLegalTypeRange(funcTy.getInputs()))
        return false;

      return isLegalTypeRange(funcTy.getResults());
    }

    return true;
  }

  bool isLegalTypeRange(const TypeRange &types) const {
    return llvm::all_of(types, [this](Type ty) { return isLegalType(ty); });
  }

private:
  unsigned maxIntWidth;
};

bool isLegalOp(Operation *op, const I64EmulationConverter &typeConverter) {
  if (!typeConverter.isLegalTypeRange(op->getOperandTypes()))
    return false;
  if (!typeConverter.isLegalTypeRange(op->getResultTypes()))
    return false;

  if (auto func = dyn_cast<func::FuncOp>(op))
    return typeConverter.isLegalType(func.getFunctionType());

  return true;
}

Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return RankedTensorType::get(tensorType.getShape(), i1Type);
  if (type.isa<UnrankedTensorType>())
    return UnrankedTensorType::get(i1Type);
  if (auto vectorType = type.dyn_cast<VectorType>())
    return VectorType::get(vectorType.getShape(), i1Type,
                           vectorType.getNumScalableDims());
  return i1Type;
}

Type peelOutermostDim(ShapedType integerLike) {
  if (auto ty = integerLike.dyn_cast<VectorType>()) {
    if (ty.getShape().size() == 1)
      return ty.getElementType();
    return VectorType::get(ty.getShape().drop_front(), ty.getElementType());
  }

  return nullptr;
}

struct ConvertAddI : OpConversionPattern<AddIOp> {
  using OpConversionPattern<AddIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddIOp op, AddIOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op->getContext();
    Location loc = op->getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto newTy = getTypeConverter()
                     ->convertType(op.getResult().getType())
                     .dyn_cast_or_null<VectorType>();
    assert(lhs.getType() == newTy);
    assert(rhs.getType() == newTy);
    Type newElemTy = peelOutermostDim(newTy);

    const int64_t idx0[1] = {0};
    const int64_t idx1[1] = {1};

    Value lhsElem0 = rewriter.create<vector::ExtractOp>(loc, lhs, idx0);
    Value lhsElem1 = rewriter.create<vector::ExtractOp>(loc, lhs, idx1);

    Value rhsElem0 = rewriter.create<vector::ExtractOp>(loc, rhs, idx0);
    Value rhsElem1 = rewriter.create<vector::ExtractOp>(loc, rhs, idx1);

    Type booleanTy = getI1SameShape(newElemTy);
    auto lowSum = rewriter.create<arith::AddICarryOp>(
        loc, newElemTy, booleanTy, lhsElem0, rhsElem0);
    Value carryVal =
        rewriter.create<arith::ExtUIOp>(loc, newElemTy, lowSum.getCarry());

    Value high0 = rewriter.create<arith::AddIOp>(loc, carryVal, lhsElem1);
    Value high = rewriter.create<arith::AddIOp>(loc, high0, rhsElem1);

    Attribute zeroAttr = SplatElementsAttr::get(newTy, APInt::getZero(32));
    Value zeroVec = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    Value vecLow =
        rewriter.create<vector::InsertOp>(loc, lowSum.getSum(), zeroVec, idx0);
    Value vecLowHigh =
        rewriter.create<vector::InsertOp>(loc, high, vecLow, idx1);
    rewriter.replaceOp(op, vecLowHigh);
    return success();
  }
};

struct EmulateI64Pass : public ArithmeticEmulateI64Base<EmulateI64Pass> {
  /// Implementation structure: first find all equivalent ops and collect them,
  /// then perform all the rewrites in a second pass over the target op. This
  /// ensures that analysis results are not invalidated during rewriting.
  void runOnOperation() override {
    llvm::errs() << "JAKUB: " << __PRETTY_FUNCTION__ << "\n";
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();

    I64EmulationConverter typeConverter(32);
    auto addUnrealizedCast = [](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) {
      auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
      return Optional<Value>(cast.getResult(0));
    };
    typeConverter.addSourceMaterialization(addUnrealizedCast);
    typeConverter.addTargetMaterialization(addUnrealizedCast);

    ConversionTarget target(*ctx);
    // clang-format off
    target.addDynamicallyLegalOp<
      // func ops
      func::FuncOp, func::CallOp, func::ReturnOp,
      // arith ops
      arith::AddIOp
    >(
    // clang-format on
        [&typeConverter](Operation *op) {
          return isLegalOp(op, typeConverter);
        });
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<vector::VectorDialect>();

    RewritePatternSet patterns(ctx);
    populateI64EmulationPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

void populateI64EmulationPatterns(TypeConverter &typeConverter,
                                  RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    ConvertAddI
   >(typeConverter, patterns.getContext());
  // clang-format on

  // Populate `func.*` conversion patterns.
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  // populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
}

std::unique_ptr<Pass> createEmulateI64Pass() {
  return std::make_unique<EmulateI64Pass>();
}

std::unique_ptr<TypeConverter>
createI64EmulationTypeConverter(unsigned int maxIntegerWidthSupported) {
  return std::make_unique<I64EmulationConverter>(maxIntegerWidthSupported);
}

} // namespace mlir::arith
