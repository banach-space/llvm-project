//===- LoopCanonicalization.cpp - Cross-dialect canonicalization patterns -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains cross-dialect canonicalization patterns that cannot be
// actual canonicalization patterns due to undesired additional dependencies.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_SCFFORLOOPCANONICALIZATION
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

using llvm::dbgs;
#define DEBUG_TYPE "loop-canon"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

/// A simple, conservative analysis to determine if the loop is shape
/// conserving. I.e., the type of the arg-th yielded value is the same as the
/// type of the corresponding basic block argument of the loop.
/// Note: This function handles only simple cases. Expand as needed.
static bool isShapePreserving(ForOp forOp, int64_t arg) {
  assert(arg < static_cast<int64_t>(forOp.getNumResults()) &&
         "arg is out of bounds");
  Value value = forOp.getYieldedValues()[arg];
  while (value) {
    if (value == forOp.getRegionIterArgs()[arg])
      return true;
    OpResult opResult = dyn_cast<OpResult>(value);
    if (!opResult)
      return false;

    using tensor::InsertSliceOp;
    value = llvm::TypeSwitch<Operation *, Value>(opResult.getOwner())
                .template Case<InsertSliceOp>(
                    [&](InsertSliceOp op) { return op.getDest(); })
                .template Case<ForOp>([&](ForOp forOp) {
                  return isShapePreserving(forOp, opResult.getResultNumber())
                             ? forOp.getInitArgs()[opResult.getResultNumber()]
                             : Value();
                })
                .Default([&](auto op) { return Value(); });
  }
  return false;
}

namespace {
/// Fold dim ops of iter_args to dim ops of their respective init args. E.g.:
///
/// ```
/// %0 = ... : tensor<?x?xf32>
/// scf.for ... iter_args(%arg0 = %0) -> (tensor<?x?xf32>) {
///   %1 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
///   ...
/// }
/// ```
///
/// is folded to:
///
/// ```
/// %0 = ... : tensor<?x?xf32>
/// scf.for ... iter_args(%arg0 = %0) -> (tensor<?x?xf32>) {
///   %1 = tensor.dim %0, %c0 : tensor<?x?xf32>
///   ...
/// }
/// ```
///
/// Note: Dim ops are folded only if it can be proven that the runtime type of
/// the iter arg does not change with loop iterations.
template <typename OpTy>
struct DimOfIterArgFolder : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy dimOp,
                                PatternRewriter &rewriter) const override {
    auto blockArg = dyn_cast<BlockArgument>(dimOp.getSource());
    if (!blockArg)
      return failure();
    auto forOp = dyn_cast<ForOp>(blockArg.getParentBlock()->getParentOp());
    if (!forOp)
      return failure();
    if (!isShapePreserving(forOp, blockArg.getArgNumber() - 1))
      return failure();

    Value initArg = forOp.getTiedLoopInit(blockArg)->get();
    rewriter.updateRootInPlace(
        dimOp, [&]() { dimOp.getSourceMutable().assign(initArg); });

    return success();
  };
};

/// Fold dim ops of loop results to dim ops of their respective init args. E.g.:
///
/// ```
/// %0 = ... : tensor<?x?xf32>
/// %r = scf.for ... iter_args(%arg0 = %0) -> (tensor<?x?xf32>) {
///   ...
/// }
/// %1 = tensor.dim %r, %c0 : tensor<?x?xf32>
/// ```
///
/// is folded to:
///
/// ```
/// %0 = ... : tensor<?x?xf32>
/// %r = scf.for ... iter_args(%arg0 = %0) -> (tensor<?x?xf32>) {
///   ...
/// }
/// %1 = tensor.dim %0, %c0 : tensor<?x?xf32>
/// ```
///
/// Note: Dim ops are folded only if it can be proven that the runtime type of
/// the iter arg does not change with loop iterations.
template <typename OpTy>
struct DimOfLoopResultFolder : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy dimOp,
                                PatternRewriter &rewriter) const override {
    auto forOp = dimOp.getSource().template getDefiningOp<scf::ForOp>();
    if (!forOp)
      return failure();
    auto opResult = cast<OpResult>(dimOp.getSource());
    unsigned resultNumber = opResult.getResultNumber();
    if (!isShapePreserving(forOp, resultNumber))
      return failure();
    rewriter.updateRootInPlace(dimOp, [&]() {
      dimOp.getSourceMutable().assign(forOp.getInitArgs()[resultNumber]);
    });
    return success();
  }
};

struct hoistRedundantVectorShapeCast : public OpRewritePattern<vector::ShapeCastOp> {
  using OpRewritePattern<vector::ShapeCastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ShapeCastOp shapeCastOp,
                                PatternRewriter &rewriter) const override {
      LLVM_DEBUG(DBGS() << "Candidate for hoisting: "
                        << *shapeCastOp.getOperation() << "\n");
      auto loop = dyn_cast<LoopLikeOpInterface>(shapeCastOp->getParentOp());
      LLVM_DEBUG(DBGS() << "Parent op: " << *shapeCastOp->getParentOp()
                        << "\n");
      if (!isa_and_nonnull<scf::ForOp, affine::AffineForOp>(loop))
        return failure();

      if (loop.getYieldedValues().size() > 1)
        return failure();

      LLVM_DEBUG(DBGS() << "Candidate shape cast: " << *shapeCastOp.getOperation()
                        << "\n");

      SetVector<Operation *> forwardSlice;
      if (!shapeCastOp.getSource().getDefiningOp())
      {
        LLVM_DEBUG(DBGS() << "Confirmed candidate shape cast: "
                          << *shapeCastOp.getOperation() << "\n");
      }

      auto candidateShape = dyn_cast<vector::ShapeCastOp>(
          loop.getYieldedValues()[0].getDefiningOp());
      if (candidateShape)
      {
        LLVM_DEBUG(DBGS() << "Double confirmed candidate shape cast: "
                          << *shapeCastOp.getOperation() << "\n");
      } else return failure();

      if (candidateShape.getSource().getType() ==
          shapeCastOp.getResult().getType()) {
        LLVM_DEBUG(DBGS() << "Triple confirmed candidate shape cast: "
                          << *shapeCastOp.getOperation() << "\n");
      }
      if (candidateShape.getResult().getType() ==
          shapeCastOp.getSource().getType()) {
        LLVM_DEBUG(DBGS() << "Quadruple confirmed candidate shape cast: "
                          << *shapeCastOp.getOperation() << "\n");
      } else return failure();

      auto initDefOp = loop.getTiedLoopInit(loop.getRegionIterArgs()[0])->get();
      loop.getLoopRegions()[0]->getBlocks();

      auto blockArg = dyn_cast<BlockArgument>(shapeCastOp.getSource());
      if (!blockArg)
        return failure();
      auto forOp = dyn_cast<ForOp>(blockArg.getParentBlock()->getParentOp());
      Value initArg = forOp.getTiedLoopInit(blockArg)->get();
      auto yieldVal = loop.getYieldedValues()[0];

      auto oldIP = rewriter.saveInsertionPoint();
      rewriter.setInsertionPoint(loop->getParentOp());
      auto newType = shapeCastOp.getResult().getType();
      auto newInit = rewriter.create<arith::ConstantOp>(
          shapeCastOp.getLoc(), newType, rewriter.getZeroAttr(newType));

      rewriter.restoreInsertionPoint(oldIP);
      auto newShapeCast = rewriter.create<vector::ShapeCastOp>(
          shapeCastOp.getLoc(), newType, yieldVal);
      // initArg.replaceAllUsesWith(newInit);
      initArg.dump();
      NewYieldValuesFn yieldFn = [&](OpBuilder &b, Location loc,
                                     ArrayRef<BlockArgument> newBBArgs) {
        return SmallVector<Value>{newShapeCast.getResult()};
      };

      auto maybeNewLoop = loop.replaceWithAdditionalYields(
          rewriter, newInit.getResult(),
          /*replaceInitOperandUsesInLoop=*/true, yieldFn);
      if (!failed(maybeNewLoop))
        maybeNewLoop->getOperation()->getParentOp()->dump();

    return success();
  }
};

/// Canonicalize AffineMinOp/AffineMaxOp operations in the context of scf.for
/// and scf.parallel loops with a known range.
template <typename OpTy>
struct AffineOpSCFCanonicalizationPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    return scf::canonicalizeMinMaxOpInLoop(rewriter, op, scf::matchForLikeLoop);
  }
};

struct SCFForLoopCanonicalization
    : public impl::SCFForLoopCanonicalizationBase<SCFForLoopCanonicalization> {
  void runOnOperation() override {
    auto *parentOp = getOperation();
    MLIRContext *ctx = parentOp->getContext();
    RewritePatternSet patterns(ctx);
    scf::populateSCFForLoopCanonicalizationPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(parentOp, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::scf::populateSCFForLoopCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns
      .add<AffineOpSCFCanonicalizationPattern<affine::AffineMinOp>,
           AffineOpSCFCanonicalizationPattern<affine::AffineMaxOp>,
           DimOfIterArgFolder<tensor::DimOp>, DimOfIterArgFolder<memref::DimOp>,
           DimOfLoopResultFolder<tensor::DimOp>,
           hoistRedundantVectorShapeCast,
           DimOfLoopResultFolder<memref::DimOp>>(ctx);
}

std::unique_ptr<Pass> mlir::createSCFForLoopCanonicalizationPass() {
  return std::make_unique<SCFForLoopCanonicalization>();
}
