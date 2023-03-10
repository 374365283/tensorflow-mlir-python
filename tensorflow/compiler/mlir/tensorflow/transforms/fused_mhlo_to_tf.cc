/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for lowering MHLO dialect to Standard dialect.

#include <utility>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace TF {
namespace {

#define GEN_PASS_DEF_FUSEDMHLOTOTFPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

class FusedMhloToTFConverter : public OpRewritePattern<mhlo::_FusedMatMulOp> {
 public:
  using OpRewritePattern<mhlo::_FusedMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::_FusedMatMulOp op, 
                                PatternRewriter &rewriter) const override {
    auto context = rewriter.getContext();
    SmallVector<Location, 3> locations{op.getLoc()};
    auto new_loc = rewriter.getFusedLoc(locations);
    Type result_type = op.getResult().getType();
    SmallVector<Value, 4> operands(op.operand_begin(),
                                   op.operand_end());
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
          NamedAttribute(StringAttr::get(context, "transpose_a"), op.getTransposeAAttr()));
    attrs.push_back(
          NamedAttribute(StringAttr::get(context, "transpose_b"), op.getTransposeBAttr()));
    
    SmallVector<Attribute, 2> fused_ops;
    auto old_fused_ops_attr = op.getFusedOpsAttr();
    for (auto fused_op_name_attr : old_fused_ops_attr) {
      auto fused_op_name = fused_op_name_attr.dyn_cast<StringAttr>().getValue();
      if (fused_op_name == "add") {
        fused_ops.push_back(StringAttr::get(context, "BiasAdd"));
      } else if (fused_op_name == "maximum") {
        fused_ops.push_back(StringAttr::get(context, "Relu"));
      }
    }

    ArrayAttr new_fused_ops_attr = ArrayAttr::get(context, fused_ops);
    attrs.push_back(
        NamedAttribute(StringAttr::get(context, "fused_ops"), new_fused_ops_attr));
    attrs.push_back(
        NamedAttribute(StringAttr::get(context, "epsilon"), op.getEpsilonAttr()));    

    Value new_op = rewriter.create<_FusedMatMulOp>(new_loc, result_type,
                                               ValueRange(operands), attrs);
    rewriter.replaceOp(op, ValueRange({new_op}));
    return success();
  }
};

struct FusedMhloToTFPass
    : public impl::FusedMhloToTFPassBase<FusedMhloToTFPass> {
  void runOnOperation() override;
};

void FusedMhloToTFPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  patterns.add<FusedMhloToTFConverter>(&getContext());

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // end anonymous namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> CreateFusedMhloToTFPass() {
  return std::make_unique<FusedMhloToTFPass>();
}

}  // end namespace TF
}  // end namespace mlir
