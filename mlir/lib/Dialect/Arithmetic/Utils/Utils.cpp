//===- Utils.cpp - Utilities to support the Linalg dialect ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;

/// Matches a ConstantIndexOp.
/// TODO: This should probably just be a general matcher that uses matchConstant
/// and checks the operation for an index type.
detail::op_matcher<arith::ConstantIndexOp> mlir::matchConstantIndex() {
  return detail::op_matcher<arith::ConstantIndexOp>();
}

/// Detects the `values` produced by a ConstantIndexOp and places the new
/// constant in place of the corresponding sentinel value.
void mlir::canonicalizeSubViewPart(
    SmallVectorImpl<OpFoldResult> &values,
    llvm::function_ref<bool(int64_t)> isDynamic) {
  for (OpFoldResult &ofr : values) {
    if (ofr.is<Attribute>())
      continue;
    // Newly static, move from Value to constant.
    if (auto cstOp =
            ofr.dyn_cast<Value>().getDefiningOp<arith::ConstantIndexOp>())
      ofr = OpBuilder(cstOp).getIndexAttr(cstOp.value());
  }
}

llvm::SmallBitVector mlir::getPositionsOfShapeOne(unsigned rank,
                                                  ArrayRef<int64_t> shape) {
  llvm::SmallBitVector dimsToProject(shape.size());
  for (unsigned pos = 0, e = shape.size(); pos < e && rank > 0; ++pos) {
    if (shape[pos] == 1) {
      dimsToProject.set(pos);
      --rank;
    }
  }
  return dimsToProject;
}

Value mlir::getValueOrCreateConstantIndexOp(OpBuilder &b, Location loc,
                                            OpFoldResult ofr) {
  if (auto value = ofr.dyn_cast<Value>())
    return value;
  auto attr = ofr.dyn_cast<Attribute>().dyn_cast<IntegerAttr>();
  assert(attr && "expect the op fold result casts to an integer attribute");
  return b.create<arith::ConstantIndexOp>(loc, attr.getValue().getSExtValue());
}

Value mlir::getValueOrCreateCastToIndexLike(OpBuilder &b, Location loc,
                                            Type targetType, Value value) {
  if (targetType == value.getType())
    return value;

  bool targetIsIndex = targetType.isIndex();
  bool valueIsIndex = value.getType().isIndex();
  if (targetIsIndex ^ valueIsIndex)
    return b.create<arith::IndexCastOp>(loc, targetType, value);

  auto targetIntegerType = targetType.dyn_cast<IntegerType>();
  auto valueIntegerType = value.getType().dyn_cast<IntegerType>();
  assert(targetIntegerType && valueIntegerType &&
         "unexpected cast between types other than integers and index");
  assert(targetIntegerType.getSignedness() == valueIntegerType.getSignedness());

  if (targetIntegerType.getWidth() > valueIntegerType.getWidth())
    return b.create<arith::ExtSIOp>(loc, targetIntegerType, value);
  return b.create<arith::TruncIOp>(loc, targetIntegerType, value);
}

SmallVector<Value>
mlir::getValueOrCreateConstantIndexOp(OpBuilder &b, Location loc,
                                      ArrayRef<OpFoldResult> valueOrAttrVec) {
  return llvm::to_vector<4>(
      llvm::map_range(valueOrAttrVec, [&](OpFoldResult value) -> Value {
        return getValueOrCreateConstantIndexOp(b, loc, value);
      }));
}

Value ArithBuilder::_and(Value lhs, Value rhs) {
  return b.create<arith::AndIOp>(loc, lhs, rhs);
}
Value ArithBuilder::add(Value lhs, Value rhs) {
  if (lhs.getType().isa<IntegerType>())
    return b.create<arith::AddIOp>(loc, lhs, rhs);
  return b.create<arith::AddFOp>(loc, lhs, rhs);
}
Value ArithBuilder::mul(Value lhs, Value rhs) {
  if (lhs.getType().isa<IntegerType>())
    return b.create<arith::MulIOp>(loc, lhs, rhs);
  return b.create<arith::MulFOp>(loc, lhs, rhs);
}
Value ArithBuilder::sgt(Value lhs, Value rhs) {
  if (lhs.getType().isa<IndexType, IntegerType>())
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, lhs, rhs);
  return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, lhs, rhs);
}
Value ArithBuilder::slt(Value lhs, Value rhs) {
  if (lhs.getType().isa<IndexType, IntegerType>())
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lhs, rhs);
  return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, lhs, rhs);
}
Value ArithBuilder::select(Value cmp, Value lhs, Value rhs) {
  return b.create<arith::SelectOp>(loc, cmp, lhs, rhs);
}

DivModValue mlir::getDivMod(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  DivModValue result;
  result.quotient = b.create<arith::DivUIOp>(loc, lhs, rhs);
  result.remainder = b.create<arith::RemUIOp>(loc, lhs, rhs);
  return result;
}

/// Create IR that computes the product of all elements in the set.
static FailureOr<OpFoldResult> getIndexProduct(OpBuilder &b, Location loc,
                                               ArrayRef<Value> set) {
  if (set.empty())
    return failure();
  OpFoldResult result = set[0];
  for (unsigned i = 1; i < set.size(); i++)
    result = b.createOrFold<arith::MulIOp>(
        loc, getValueOrCreateConstantIndexOp(b, loc, result), set[i]);
  return result;
}

FailureOr<SmallVector<Value>> mlir::delinearizeIndex(OpBuilder &b, Location loc,
                                                     Value linearIndex,
                                                     ArrayRef<Value> dimSizes) {
  unsigned numDims = dimSizes.size();

  SmallVector<Value> divisors;
  for (unsigned i = 1; i < numDims; i++) {
    ArrayRef<Value> slice(dimSizes.begin() + i, dimSizes.end());
    FailureOr<OpFoldResult> prod = getIndexProduct(b, loc, slice);
    if (failed(prod))
      return failure();
    divisors.push_back(getValueOrCreateConstantIndexOp(b, loc, *prod));
  }

  SmallVector<Value> results;
  Value residual = linearIndex;
  for (Value divisor : divisors) {
    DivModValue divMod = getDivMod(b, loc, residual, divisor);
    results.push_back(divMod.quotient);
    residual = divMod.remainder;
  }
  results.push_back(residual);
  return results;
}
