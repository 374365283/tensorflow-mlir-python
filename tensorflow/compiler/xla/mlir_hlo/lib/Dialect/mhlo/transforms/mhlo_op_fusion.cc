/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdio>
#include <iostream>
#include <string>
#include <utility>
#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_MHLOOPFUSIONPASS
#include "mlir-hlo/Dialect/mhlo/transforms/mhlo_passes.h.inc"

namespace {

Operation *GetAdd(Value op) {
  for (auto &use : op.getUses()) {
    if (isa<AddOp>(use.getOwner())) return use.getOwner();
  }
  return nullptr;
}

Operation *GetMax(Value op) {
  for (auto &use : op.getUses()) {
    if (isa<MaxOp>(use.getOwner())) return use.getOwner();
  }
  return nullptr;
}

class FuseDotAddMaxPattern : public OpRewritePattern<DotOp> {
  public:
    using OpRewritePattern<DotOp>::OpRewritePattern;
    bool AreFuseCompatible(DotOp dot_op, AddOp add_op,
                                 PatternRewriter &rewriter) const {
      return true;
    }

    bool IsDeviceCompatible(DotOp dot_op, AddOp add_op,
                                  PatternRewriter &rewriter) const {
      return true;
    } 

    LogicalResult matchAndRewrite(DotOp dot_op,
                                PatternRewriter &rewriter) const override {
      auto context = rewriter.getContext();

      if (!isa<func::FuncOp, IfOp, WhileOp>(dot_op->getParentOp())) {
        return rewriter.notifyMatchFailure(
            dot_op,
            "fused operation must be nested inside a function, If or While");
      }
      
      if (!dot_op.getResult().hasOneUse())
        return rewriter.notifyMatchFailure(dot_op,
                                         "result is used by multiple ops");

      Operation* add_operation = GetAdd(dot_op.getResult());
      if (!add_operation) {
        return rewriter.notifyMatchFailure(
            dot_op, "does not feed into a mhlo.add op");
      }
      AddOp add_op = dyn_cast_or_null<AddOp>(add_operation);

      if (!AreFuseCompatible(dot_op, add_op, rewriter)) {
        return rewriter.notifyMatchFailure(
            dot_op, "cannot fuse with the subsequent Add op");
      }

      if (!IsDeviceCompatible(dot_op, add_op, rewriter)) {
        return rewriter.notifyMatchFailure(
            dot_op,
            "cannot fuse with the subsequent op as it's not supported by the "
            "target device.");
      }

      SmallVector<Location, 3> locations{dot_op.getLoc(), add_op.getLoc()};
      SmallVector<Attribute, 2> fused_ops{StringAttr::get(
        context, add_op.getOperation()->getName().stripDialect())};
      SmallVector<Value, 4> operands;
      std::vector<NamedAttribute> attrs = dot_op->getAttrs();
      
      Type result_type = add_op.getResult().getType();
      
      Operation* max_operation = GetMax(add_op.getResult());
      bool fuse_max = max_operation && add_op.getResult().hasOneUse();
      
      Value dot_left = dot_op.getLhs();
      Value dot_right = dot_op.getRhs();
      auto transpose_a_op = dot_left.getDefiningOp<TransposeOp>();
      auto transpose_b_op = dot_right.getDefiningOp<TransposeOp>();
      bool fuse_transpose_a = transpose_a_op && transpose_a_op.getResult().hasOneUse();
      bool fuse_transpose_b = transpose_b_op && transpose_b_op.getResult().hasOneUse();
      if (fuse_transpose_a) {
        locations.push_back(transpose_a_op.getLoc());
        operands.push_back(*(transpose_a_op.getODSOperands(0).begin()));
      } else {
        operands.push_back(*(dot_op.getODSOperands(0).begin()));
      }
      attrs.push_back(
          NamedAttribute(StringAttr::get(context, "transpose_a"), BoolAttr::get(context, fuse_transpose_a)));

      if (fuse_transpose_b) {
        locations.push_back(transpose_b_op.getLoc());
        operands.push_back(*(transpose_b_op.getODSOperands(0).begin()));
      } else {
        operands.push_back(*(dot_op.getODSOperands(1).begin()));
      }
      attrs.push_back(
          NamedAttribute(StringAttr::get(context, "transpose_b"), BoolAttr::get(context, fuse_transpose_b)));

      Value add_left = add_op.getLhs();
      auto broadcast_in_dim_op = add_left.getDefiningOp<BroadcastInDimOp>();
      if (!broadcast_in_dim_op) {
        Value add_right = add_op.getRhs();
        broadcast_in_dim_op = add_right.getDefiningOp<BroadcastInDimOp>();
        if (!broadcast_in_dim_op) {
          return rewriter.notifyMatchFailure(
            add_op,
            "cannot fuse with the subsequent op as two inputs of add op is not dot and broadcast_in_dim op.");
        }
      }

      if (!broadcast_in_dim_op.getResult().hasOneUse())
        return rewriter.notifyMatchFailure(broadcast_in_dim_op, "result is used by multiple ops");

      locations.push_back(broadcast_in_dim_op.getLoc());
      operands.push_back(*(broadcast_in_dim_op.getODSOperands(0).begin()));

      bool left_is_add = false;
      Operation* constant_operation = nullptr;
      if (fuse_max) {
        auto max_op = dyn_cast_or_null<MaxOp>(max_operation);
        Value max_left = max_op.getLhs();
        auto constant_op = max_left.getDefiningOp<ConstantOp>();
        if (!constant_op) {
          Value max_right = max_op.getRhs();
          constant_op = max_right.getDefiningOp<ConstantOp>();
          if (!constant_op) {
            return rewriter.notifyMatchFailure(
              max_op,
              "cannot fuse with the subsequent op as two inputs of max op is not add and constant op.");
          }
        }
        locations.push_back(max_op.getLoc());
        fused_ops.push_back(
          StringAttr::get(context, max_operation->getName().stripDialect()));
        result_type = max_op.getResult().getType();

        if (constant_op.getResult().hasOneUse()) {
          locations.push_back(constant_op.getLoc());
        }
      }

      ArrayAttr fused_ops_attr = ArrayAttr::get(context, fused_ops);
      attrs.push_back(
        NamedAttribute(StringAttr::get(context, "fused_ops"), fused_ops_attr));

      // Epsilon is used only in fusions with the FusedBatchNorm op, so we zero it
      // here.
      Attribute epsilon = rewriter.getF32FloatAttr(0);
      attrs.push_back(
        NamedAttribute(StringAttr::get(context, "epsilon"), epsilon));

      rewriter.setInsertionPoint(add_op);
      auto fused_loc = rewriter.getFusedLoc(locations);
      Value fused_op = rewriter.create<_FusedMatMulOp>(fused_loc, result_type,
                                               ValueRange(operands), attrs);
      auto op_to_replace = fuse_max ? max_operation : add_op;
      rewriter.replaceOp(op_to_replace, ValueRange({fused_op}));
      return success();
    }
};

struct MhloOpFusionPass
    : public impl::MhloOpFusionPassBase<MhloOpFusionPass> {
  void runOnOperation() override;
};

void MhloOpFusionPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  patterns.add<FuseDotAddMaxPattern>(&getContext());
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createMhloOpFusionPass() {
  return std::make_unique<MhloOpFusionPass>();
}

}  // namespace mhlo
}  // namespace mlir
