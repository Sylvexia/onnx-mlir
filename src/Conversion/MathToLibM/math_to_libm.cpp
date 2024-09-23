#include "src/Conversion/MathToLibM/math_to_libm.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Pass/Passes.hpp"
#include <memory>

using namespace mlir;

struct ConvertCustomMathToLibmPass
    : public PassWrapper<ConvertCustomMathToLibmPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
  StringRef getArgument() const override;
  StringRef getDescription() const override;
};

void ConvertCustomMathToLibmPass::runOnOperation() {
  auto module = getOperation();
  RewritePatternSet patterns(&getContext());
  populateCustomMathToLibmConversionPatterns(patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect>();
  target.addIllegalDialect<math::MathDialect>();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

StringRef ConvertCustomMathToLibmPass::getArgument() const {
  return "convert-custom-math-to-llvm";
}

StringRef ConvertCustomMathToLibmPass::getDescription() const {
  return "Lower the Krnl Affine and Std dialects to LLVM.";
}

template <typename Op>
struct CustomMathOpToLibmCall : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;
  CustomMathOpToLibmCall(
      MLIRContext *context, StringRef floatref, StringRef doubleref)
      : mlir::OpRewritePattern<Op>(context), floatFuncName(floatref),
        doubleFuncName(doubleref){};

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;

private:
  std::string floatFuncName, doubleFuncName;
};

template <typename Op>
LogicalResult CustomMathOpToLibmCall<Op>::matchAndRewrite(
    Op op, PatternRewriter &rewriter) const {
  auto module = SymbolTable::getNearestSymbolTable(op);
  auto type = op.getType();
  if (!isa<Float32Type, Float64Type>(type))
    return failure();

  auto name =
      type.getIntOrFloatBitWidth() == 64 ? doubleFuncName : floatFuncName;
  auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
    SymbolTable::lookupSymbolIn(module, name));
  if(!opFunc){
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&module->getRegion(0).front());
    auto opFunctionTy = FunctionType::get(
      rewriter.getContext(), op->getOperandTypes(), op->getResultTypes());
    opFunc = rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(), name,
    opFunctionTy); opFunc.setPrivate();
    opFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(),
                    UnitAttr::get(rewriter.getContext()));
  }
  assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module,
  name)));

  rewriter.replaceOpWithNewOp<func::CallOp>(
      op, name, op.getType(), op->getOperands());
  return success();
}

// struct ExpOpLowering : public OpRewritePattern<mlir::math::ExpOp> {
//   using OpRewritePattern<mlir::math::ExpOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(mlir::math::ExpOp expOp, PatternRewriter
//   &rewriter) const override {
//     auto loc = expOp.getLoc();

//     auto operand = expOp.getOperand();
//     auto type = operand.getType();

//     FlatSymbolRefAttr libmFunc;
//     if (type.isF32()) {
//       libmFunc = SymbolRefAttr::get(rewriter.getContext(), "expf");
//     } else if (type.isF64()) {
//       libmFunc = SymbolRefAttr::get(rewriter.getContext(), "exp");
//     } else {
//       return failure();
//     }

//     auto callOp = rewriter.create<LLVM::LLVMFuncOp>(loc, libmFunc, type,
//     operand);

//     // Replace the original exp operation with the call
//     // rewriter.replaceOp(expOp, callOp.getResult());

//     return success();
//   }
// };

template <typename OpTy>
void populatePatternsForOp(RewritePatternSet &patterns, MLIRContext *ctx,
    StringRef floatFunc, StringRef doubleFunc) {
  patterns.add<CustomMathOpToLibmCall<OpTy>>(ctx, floatFunc, doubleFunc);
}

void mlir::populateCustomMathToLibmConversionPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  populatePatternsForOp<math::ExpOp>(patterns, ctx, "expf", "exp");
  populatePatternsForOp<math::Exp2Op>(patterns, ctx, "exp2f", "exp2");
}

std::unique_ptr<Pass> mlir::createConvertCustomMathToLibmPass() {
  return std::make_unique<ConvertCustomMathToLibmPass>();
}