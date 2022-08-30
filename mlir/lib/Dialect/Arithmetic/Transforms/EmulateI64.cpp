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
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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
      const unsigned width = ty.getWidth();
      if (width <= widestInt)
        return ty;

      if (width == 2 * widestInt)
        return VectorType::get({2},
                               IntegerType::get(ty.getContext(), widestInt));

      return None;
    });

    // Vector case.
    addConversion([widestInt = maxIntegerWidthSupported](
                      VectorType ty) -> Optional<Type> {
      if (auto intTy = ty.getElementType().dyn_cast<IntegerType>()) {
        const unsigned width = intTy.getWidth();
        if (width <= widestInt)
          return ty;

        if (width == 2 * widestInt) {
          SmallVector<int64_t> newShape = {2};
          llvm::append_range(newShape, ty.getShape());
          return VectorType::get(newShape,
                                 IntegerType::get(ty.getContext(), widestInt));
        }

        return None;
      }
      return ty;
    });

    // Function case.
    addConversion([this](FunctionType ty) -> Optional<Type> {
      SmallVector<Type> inputs;
      if (failed(convertTypes(ty.getInputs(), inputs)))
        return None;

      SmallVector<Type> results;
      if (failed(convertTypes(ty.getResults(), results)))
        return None;

      return FunctionType::get(ty.getContext(), inputs, results);
    });
  }

  unsigned getMaxIntegerWidth() const { return maxIntWidth; }

private:
  unsigned maxIntWidth;
};

Type peelOutermostDim(ShapedType integerLike) {
  if (auto ty = integerLike.dyn_cast<VectorType>()) {
    if (ty.getShape().size() == 1)
      return ty.getElementType();
    return VectorType::get(ty.getShape().drop_front(), ty.getElementType());
  }

  return nullptr;
}

struct ConvertAddI : OpConversionPattern<AddIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddIOp op, AddIOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto &typeConverter = *getTypeConverter<I64EmulationConverter>();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto newTy = typeConverter.convertType(op.getResult().getType())
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

    auto lowSum = rewriter.create<arith::AddUICarryOp>(loc, lhsElem0, rhsElem0);
    Value carryVal =
        rewriter.create<arith::ExtUIOp>(loc, newElemTy, lowSum.getCarry());

    Value high0 = rewriter.create<arith::AddIOp>(loc, carryVal, lhsElem1);
    Value high = rewriter.create<arith::AddIOp>(loc, high0, rhsElem1);

    Attribute zeroAttr = SplatElementsAttr::get(
        newTy, APInt::getZero(typeConverter.getMaxIntegerWidth()));
    Value zeroVec = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    Value vecLow =
        rewriter.create<vector::InsertOp>(loc, lowSum.getSum(), zeroVec, idx0);
    Value vecLowHigh =
        rewriter.create<vector::InsertOp>(loc, high, vecLow, idx1);
    rewriter.replaceOp(op, vecLowHigh);
    return success();
  }
};

struct ConvertConstant : OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, ConstantOpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TypeConverter &typeConverter = *getTypeConverter();

    Type oldType = op.getType();
    auto newType = typeConverter.convertType(oldType).cast<ShapedType>();
    const unsigned newBitWidth = newType.getElementTypeBitWidth();
    Attribute oldValue = op.getValueAttr();

    if (auto intAttr = oldValue.dyn_cast<IntegerAttr>()) {
      auto [low, high] = getHalves(intAttr.getValue(), newBitWidth);
      auto newAttr = DenseElementsAttr::get(newType, {low, high});
      rewriter.replaceOpWithNewOp<ConstantOp>(op, newAttr);
      return success();
    }

    if (auto splatAttr = oldValue.dyn_cast<SplatElementsAttr>()) {
      auto [low, high] =
          getHalves(splatAttr.getSplatValue<APInt>(), newBitWidth);
      const auto numSplatElems =
          static_cast<size_t>(splatAttr.getNumElements());
      auto values = llvm::to_vector(
          llvm::concat<APInt>(SmallVector<APInt>(numSplatElems, low),
                              SmallVector<APInt>(numSplatElems, high)));

      auto attr = DenseElementsAttr::get(newType, values);
      rewriter.replaceOpWithNewOp<ConstantOp>(op, attr);
      return success();
    }

    if (auto elemsAttr = oldValue.dyn_cast<DenseElementsAttr>()) {
      const auto numElems = static_cast<size_t>(elemsAttr.getNumElements());
      SmallVector<APInt> lowVals;
      lowVals.reserve(numElems);
      SmallVector<APInt> highVals;
      highVals.reserve(numElems);

      for (const APInt &origVal : elemsAttr.getValues<APInt>()) {
        auto [low, high] = getHalves(origVal, newBitWidth);
        lowVals.push_back(std::move(low));
        highVals.push_back(std::move(high));
      }
      auto values = llvm::to_vector(
          llvm::concat<APInt>(std::move(lowVals), std::move(highVals)));

      auto attr = DenseElementsAttr::get(newType, values);
      rewriter.replaceOpWithNewOp<ConstantOp>(op, attr);
      return success();
    }

    return failure();
  }

private:
  static std::pair<APInt, APInt> getHalves(const APInt &value,
                                           unsigned newBitWidth) {
    APInt low = value.extractBits(newBitWidth, 0);
    APInt high = value.extractBits(newBitWidth, newBitWidth);
    return {std::move(low), std::move(high)};
  }
};

struct ConvertExtSI : OpConversionPattern<ExtSIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtSIOp op, ExtSIOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &typeConverter = *getTypeConverter<I64EmulationConverter>();
    if (!typeConverter.isLegal(op.getIn().getType()))
      return failure();

    Location loc = op->getLoc();

    Type oldTy = op.getType();
    auto newTy = typeConverter.convertType(oldTy).cast<ShapedType>();
    Type newOperandTy = peelOutermostDim(newTy);
    const unsigned newBitWidth = newTy.getElementTypeBitWidth();

    Value extended =
        rewriter.createOrFold<ExtSIOp>(loc, newOperandTy, adaptor.getIn());
    Attribute operandZeroAttr;
    if (newOperandTy.isa<IntegerType>())
      operandZeroAttr =
          IntegerAttr::get(newOperandTy, APInt::getZero(newBitWidth));
    else
      operandZeroAttr =
          DenseElementsAttr::get(newOperandTy, APInt::getZero(newBitWidth));

    Value operandZeroCst = rewriter.create<ConstantOp>(loc, operandZeroAttr);
    Value signBit = rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, extended,
                                            operandZeroCst);
    Value signValue = rewriter.create<ExtSIOp>(loc, newOperandTy, signBit);

    Value vecZeroCst = rewriter.create<ConstantOp>(
        loc, DenseElementsAttr::get(newTy, APInt::getZero(newBitWidth)));
    Value ins0 = rewriter.create<vector::InsertOp>(
        loc, extended, vecZeroCst, llvm::makeArrayRef(int64_t(0)));
    rewriter.replaceOpWithNewOp<vector::InsertOp>(
        op, signValue, ins0, llvm::makeArrayRef(int64_t(1)));
    return success();
  }
};

struct ConvertExtUI : OpConversionPattern<ExtUIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtUIOp op, ExtUIOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &typeConverter = *getTypeConverter<I64EmulationConverter>();
    if (!typeConverter.isLegal(op.getIn().getType()))
      return failure();

    Type oldTy = op.getType();
    auto newTy = typeConverter.convertType(oldTy).cast<ShapedType>();
    Type newOperandTy = peelOutermostDim(newTy);
    const unsigned newBitWidth = newTy.getElementTypeBitWidth();

    Value extended = rewriter.createOrFold<ExtUIOp>(op->getLoc(), newOperandTy,
                                                    adaptor.getIn());
    Attribute zeroAttr =
        DenseElementsAttr::get(newTy, APInt::getZero(newBitWidth));
    Value zeroCst = rewriter.create<ConstantOp>(op->getLoc(), zeroAttr);
    rewriter.replaceOpWithNewOp<vector::InsertOp>(
        op, extended, zeroCst, llvm::makeArrayRef(int64_t(0)));
    return success();
  }
};

struct ConvertMulI : OpConversionPattern<MulIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MulIOp op, MulIOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto &typeConverter = *getTypeConverter<I64EmulationConverter>();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto newTy = typeConverter.convertType(op.getResult().getType())
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

    auto lowSum = rewriter.create<arith::AddUICarryOp>(loc, lhsElem0, rhsElem0);
    Value carryVal =
        rewriter.create<arith::ExtUIOp>(loc, newElemTy, lowSum.getCarry());

    Value high0 = rewriter.create<arith::AddIOp>(loc, carryVal, lhsElem1);
    Value high = rewriter.create<arith::AddIOp>(loc, high0, rhsElem1);

    Attribute zeroAttr = SplatElementsAttr::get(
        newTy, APInt::getZero(typeConverter.getMaxIntegerWidth()));
    Value zeroVec = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    Value vecLow =
        rewriter.create<vector::InsertOp>(loc, lowSum.getSum(), zeroVec, idx0);
    Value vecLowHigh =
        rewriter.create<vector::InsertOp>(loc, high, vecLow, idx1);
    rewriter.replaceOp(op, vecLowHigh);
    return success();
  }
};

struct ConvertTruncI : OpConversionPattern<TruncIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TruncIOp op, TruncIOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &typeConverter = *getTypeConverter<I64EmulationConverter>();
    if (!typeConverter.isLegal(op.getType()))
      return failure();

    Value extracted = rewriter.create<vector::ExtractOp>(
        op->getLoc(), adaptor.getIn(), llvm::makeArrayRef(int64_t(0)));
    Value truncated =
        rewriter.createOrFold<TruncIOp>(op->getLoc(), op.getType(), extracted);
    rewriter.replaceOp(op, truncated);
    return success();
  }
};

struct EmulateI64Pass : public ArithmeticEmulateI64Base<EmulateI64Pass> {
  /// Implementation structure: first find all equivalent ops and collect them,
  /// then perform all the rewrites in a second pass over the target op. This
  /// ensures that analysis results are not invalidated during rewriting.
  void runOnOperation() override {
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
      arith::ConstantOp,
      arith::AddIOp, arith::MulIOp,
      arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp
    >(
        // clang-format on
        [&typeConverter](Operation *op) {
          if (auto func = dyn_cast<func::FuncOp>(op))
            return typeConverter.isLegal(func.getFunctionType());

          return typeConverter.isLegal(op);
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
    ConvertConstant,
    ConvertAddI, ConvertMulI,
    ConvertExtSI, ConvertExtUI, ConvertTruncI
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
