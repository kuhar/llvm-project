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
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <cassert>

namespace mlir::arith {
#define GEN_PASS_DEF_ARITHMETICEMULATEWIDEINT
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h.inc"
} // namespace mlir::arith

using namespace mlir;

namespace {
// Converts integer types that are too wide for the target to supported ones.
// Currently, we only handle power-of-two integer types and support conversions
// of integers twice as wide as the maxium supported. Wide integers are
// represented as vectors, e.g., i64 --> vector<2xi32>, where the first element
// is the low half of the original integer, and the second element the high
// half.
class WideIntEmulationConverter final : public TypeConverter {
public:
  explicit WideIntEmulationConverter(unsigned widestIntSupported)
      : maxIntWidth(widestIntSupported) {
    // Scalar case.
    addConversion([widestInt =
                       widestIntSupported](IntegerType ty) -> Optional<Type> {
      const unsigned width = ty.getWidth();
      if (width <= widestInt)
        return ty;

      // i2N --> vector<2xiN>
      if (width == 2 * widestInt)
        return VectorType::get({2},
                               IntegerType::get(ty.getContext(), widestInt));

      return None;
    });

    // Vector case.
    addConversion([widestInt =
                       widestIntSupported](VectorType ty) -> Optional<Type> {
      if (auto intTy = ty.getElementType().dyn_cast<IntegerType>()) {
        const unsigned width = intTy.getWidth();
        if (width <= widestInt)
          return ty;

        // vector<...xi2N> --> vector<2x...xiN>
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
      // Convert inputs and results, e.g.:
      //   (i2N, i2N) -> i2N --> (vector<2xiN>, vector<2xiN>) -> vector<2xiN>
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

struct ConvertConstant final : OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type oldType = op.getType();
    auto newType = getTypeConverter()->convertType(oldType).cast<ShapedType>();
    const unsigned newBitWidth = newType.getElementTypeBitWidth();
    Attribute oldValue = op.getValueAttr();

    if (auto intAttr = oldValue.dyn_cast<IntegerAttr>()) {
      auto [low, high] = getHalves(intAttr.getValue(), newBitWidth);
      auto newAttr = DenseElementsAttr::get(newType, {low, high});
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newAttr);
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
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, attr);
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
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, attr);
      return success();
    }

    return rewriter.notifyMatchFailure(op.getLoc(),
                                       "Unhandled constant attribute");
  }

private:
  static std::pair<APInt, APInt> getHalves(const APInt &value,
                                           unsigned newBitWidth) {
    APInt low = value.extractBits(newBitWidth, 0);
    APInt high = value.extractBits(newBitWidth, newBitWidth);
    return {std::move(low), std::move(high)};
  }
};

struct EmulateWideIntPass final
    : arith::impl::ArithmeticEmulateWideIntBase<EmulateWideIntPass> {
  EmulateWideIntPass(unsigned widestIntSupported) {
    this->widestIntSupported.setValue(widestIntSupported);
  }

  /// Implementation structure: first find all equivalent ops and collect them,
  /// then perform all the rewrites in a second pass over the target op. This
  /// ensures that analysis results are not invalidated during rewriting.
  void runOnOperation() override {
    if (!llvm::isPowerOf2_32(widestIntSupported)) {
      assert(false && "Widest int supported is not a power of two");
      signalPassFailure();
      return;
    }

    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();

    WideIntEmulationConverter typeConverter(widestIntSupported);
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
      // arith ops
      arith::ConstantOp,
      // func ops
      func::FuncOp, func::CallOp, func::ReturnOp
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
    arith::populateWideIntEmulationPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // end anonymous namespace

namespace mlir::arith {

void populateWideIntEmulationPatterns(TypeConverter &typeConverter,
                                      RewritePatternSet &patterns) {
  patterns.add<ConvertConstant>(typeConverter, patterns.getContext());

  // Populate `func.*` conversion patterns.
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
}

std::unique_ptr<Pass> createEmulateWideIntPass(unsigned widestIntSupported) {
  return std::make_unique<EmulateWideIntPass>(widestIntSupported);
}

std::unique_ptr<TypeConverter>
createWideIntEmulationTypeConverter(unsigned widestIntSupported) {
  return std::make_unique<WideIntEmulationConverter>(widestIntSupported);
}

} // namespace mlir::arith
