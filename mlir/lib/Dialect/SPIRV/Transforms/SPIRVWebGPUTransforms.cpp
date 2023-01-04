//===- SPIRVWebGPUTransforms.cpp - WebGPU-specific transforms -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements SPIR-V transforms used when targetting WebGPU.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Transforms/SPIRVWebGPUTransforms.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace mlir {
namespace spirv {
#define GEN_PASS_DEF_SPIRVWEBGPUPREPAREPASS
#include "mlir/Dialect/SPIRV/Transforms/Passes.h.inc"
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
namespace {
//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//
Attribute getScalarOrSplatAttr(Type type, int64_t value) {
  APInt sizedValue(getElementTypeOrSelf(type).getIntOrFloatBitWidth(), value);
  if (auto intTy = type.dyn_cast<IntegerType>())
    return IntegerAttr::get(intTy, sizedValue);

  return SplatElementsAttr::get(type, sizedValue);
}

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//
struct ExpandSMulExtendedPattern final : OpRewritePattern<SMulExtendedOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SMulExtendedOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value lhs = op.getOperand1();
    Value rhs = op.getOperand2();
    Type argTy = lhs.getType();

    // Currently, WGSL only supports 32-bit integer types. Any other integer
    // types should already have been promoted/demoted to i32.
    auto elemTy = getElementTypeOrSelf(argTy).cast<IntegerType>();
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return rewriter.notifyMatchFailure(
          loc,
          llvm::formatv("Unexpected integer type for WebGPU: '{0}'", elemTy));

    // Calculate the 'low' and the 'high' result separately. Use UMulExtended
    // expansion so that we only have to worry about multiplication with
    // sign-extended bits.
    //
    // lhs = [s0   s0]  [a   b]
    // rhs = [s1   s1]  [c   d]
    // --lhs * rhs--
    // =     [0   0][a  b] * [0   0][c   d] +
    //       [s0 s0][0  0] * [0   0][c   d] +
    //       [0   0][a  b] * [s1 s1][c   d]
    //
    // ==> high = UMulExtended(lhs, rhs).high +          ; high0 +
    //            (s0 * d) + (s1 * b) +                  ; high1 + high2 +
    //           ((s0 * c) + (s1 * a)) >> 16             ; high3
    Value cst16 = rewriter.create<ConstantOp>(loc, argTy,
                                              getScalarOrSplatAttr(argTy, 16));
    Value cst31 = rewriter.create<ConstantOp>(loc, argTy,
                                              getScalarOrSplatAttr(argTy, 31));
    Value cstLowMask = rewriter.create<ConstantOp>(
        loc, argTy, getScalarOrSplatAttr(argTy, (1 << 16) - 1));
    auto getLowHalf = [&rewriter, loc, cstLowMask](Value val) {
      return rewriter.create<BitwiseAndOp>(loc, val, cstLowMask);
    };
    auto getHighHalf = [&rewriter, loc, cst16](Value val) {
      return rewriter.create<ShiftRightLogicalOp>(loc, val, cst16);
    };
    Value lhsLow = getLowHalf(lhs);
    Value lhsHigh = getHighHalf(lhs);
    Value rhsLow = getLowHalf(rhs);
    Value rhsHigh = getHighHalf(rhs);

    // This extended multiplication will be expanded by another pattern.
    Value umul = rewriter.create<UMulExtendedOp>(loc, lhs, rhs);
    Value low =
        rewriter.create<CompositeExtractOp>(loc, umul, llvm::makeArrayRef(0));
    Value high0 =
        rewriter.create<CompositeExtractOp>(loc, umul, llvm::makeArrayRef(1));

    // The sign extensions of op arguments.
    Value lhsSign = rewriter.create<ShiftRightArithmeticOp>(loc, lhs, cst31);
    Value rhsSign = rewriter.create<ShiftRightArithmeticOp>(loc, rhs, cst31);

    Value high1 = rewriter.create<IMulOp>(loc, lhsSign, rhsLow);
    Value high2 = rewriter.create<IMulOp>(loc, rhsSign, lhsLow);

    Value tmp0 = rewriter.create<IMulOp>(loc, lhsSign, rhsHigh);
    Value tmp1 = rewriter.create<IMulOp>(loc, rhsSign, lhsHigh);
    Value high3 = getHighHalf(rewriter.create<IAddOp>(loc, tmp0, tmp1));

    Value high =
        rewriter.create<IAddOp>(loc, rewriter.create<IAddOp>(loc, high0, high1),
                                rewriter.create<IAddOp>(loc, high2, high3));
    rewriter.replaceOpWithNewOp<CompositeConstructOp>(
        op, op.getType(), llvm::makeArrayRef({low, high}));
    return success();
  }
};

struct ExpandUMulExtendedPattern final : OpRewritePattern<UMulExtendedOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UMulExtendedOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value lhs = op.getOperand1();
    Value rhs = op.getOperand2();
    Type argTy = lhs.getType();

    // Currently, WGSL only supports 32-bit integer types. Any other integer
    // types should already have been promoted/demoted to i32.
    auto elemTy = getElementTypeOrSelf(argTy).cast<IntegerType>();
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return rewriter.notifyMatchFailure(
          loc,
          llvm::formatv("Unexpected integer type for WebGPU: '{0}'", elemTy));

    // Calculate the 'low' and the 'high' result separately, using long
    // multiplication:
    //
    // lhs = [0   0]  [a   b]
    // rhs = [0   0]  [c   d]
    // --lhs * rhs--
    // =     [    a * c    ]   [    b * d    ] +
    //       [ 0 ]    [a * d + b * c]    [ 0 ]
    //
    // ==> high = (a * c) + (a * d + b * c) >> 16
    Value low = rewriter.create<IMulOp>(loc, lhs, rhs);

    Value cstLowMask = rewriter.create<ConstantOp>(
        loc, argTy, getScalarOrSplatAttr(argTy, (1 << 16) - 1));
    auto getLowHalf = [&rewriter, loc, cstLowMask](Value val) {
      return rewriter.create<BitwiseAndOp>(loc, val, cstLowMask);
    };

    Value cst16 = rewriter.create<ConstantOp>(loc, argTy,
                                              getScalarOrSplatAttr(argTy, 16));
    auto getHighHalf = [&rewriter, loc, cst16](Value val) {
      return rewriter.create<ShiftRightLogicalOp>(loc, val, cst16);
    };

    Value lhsLow = getLowHalf(lhs);
    Value lhsHigh = getHighHalf(lhs);
    Value rhsLow = getLowHalf(rhs);
    Value rhsHigh = getHighHalf(rhs);

    Value high0 = rewriter.create<IMulOp>(loc, lhsHigh, rhsHigh);
    Value mid = rewriter.create<IAddOp>(
        loc, rewriter.create<IMulOp>(loc, lhsHigh, rhsLow),
        rewriter.create<IMulOp>(loc, lhsLow, rhsHigh));
    Value high1 = getHighHalf(mid);
    Value high = rewriter.create<IAddOp>(loc, high0, high1);

    rewriter.replaceOpWithNewOp<CompositeConstructOp>(
        op, op.getType(), llvm::makeArrayRef({low, high}));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//
class WebGPUPreparePass
    : public impl::SPIRVWebGPUPreparePassBase<WebGPUPreparePass> {
public:
  void runOnOperation() override {
    llvm::errs() << __PRETTY_FUNCTION__ << "\n";
    RewritePatternSet patterns(&getContext());
    populateSPIRVExpandExtendedMultiplicationPatterns(patterns);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Public Interface
//===----------------------------------------------------------------------===//
void populateSPIRVExpandExtendedMultiplicationPatterns(
    RewritePatternSet &patterns) {
  // WGSL currently does not support extended multiplication ops, see:
  // https://github.com/gpuweb/gpuweb/issues/1565.
  patterns.add<ExpandSMulExtendedPattern, ExpandUMulExtendedPattern>(
      patterns.getContext());
}
} // namespace spirv
} // namespace mlir
