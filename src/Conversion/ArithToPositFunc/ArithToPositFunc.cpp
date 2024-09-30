#include "src/Conversion/ArithToPositFunc/ArithToPositFunc.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Pass/Passes.hpp"
#include <sys/types.h>

using namespace mlir;

struct ConvertArithToPositFuncPass
    : public PassWrapper<ConvertArithToPositFuncPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
  StringRef getArgument() const override;
  StringRef getDescription() const override;
};

void ConvertArithToPositFuncPass::runOnOperation() {
  auto module = getOperation();
  RewritePatternSet patterns(&getContext());
  mlir::populateArithToPositFuncConversionPatterns(patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect>();
  target.addIllegalDialect<arith::ArithDialect>();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

StringRef ConvertArithToPositFuncPass::getArgument() const {
  return "convert-arith-to-posit-func";
}

StringRef ConvertArithToPositFuncPass::getDescription() const {
  return "Lower the arith dialect to posit func dialect.";
}

template <typename Op>
struct ArithToPositFuncLowering : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;
  ArithToPositFuncLowering(
      MLIRContext *context, StringRef opString, uint8_t n_bits, uint8_t es_val)
      : mlir::OpRewritePattern<Op>(context), opString(opString), n_bits(n_bits),
        es_val(es_val){};

  LogicalResult matchAndRewrite(Op op, PatternRewriter &Rewriter) const final;

private:
  std::string opString;
  uint8_t n_bits;
  uint8_t es_val;
};

template <typename Op>
LogicalResult ArithToPositFuncLowering<Op>::matchAndRewrite(
    Op op, PatternRewriter &rewriter) const {
  auto symbolTable = SymbolTable::getNearestSymbolTable(op);
  auto type = op.getType();
  auto context = rewriter.getContext();

  if (!isa<Float32Type>(type))
    return failure();

  // e.g. posit8es0_add
  std::string name = "posit" + std::to_string(n_bits) + "es" +
                     std::to_string(es_val) + "_" + opString;

  uint8_t numOperands = 2;
  uint8_t numResults = 1;

  Type ConvertedType =
      mlir::IntegerType::get(context, n_bits, IntegerType::Unsigned);

  std::vector<Type> operandTypesVec;
  for (uint8_t i = 0; i < numOperands; i++) {
    operandTypesVec.push_back(ConvertedType);
  }
  std::vector<Type> resultTypesVec;
  for (uint8_t i = 0; i < numResults; i++) {
    resultTypesVec.push_back(ConvertedType);
  }

  TypeRange operandTypes{operandTypesVec};
  TypeRange resultTypes{resultTypesVec};

  auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
      SymbolTable::lookupSymbolIn(symbolTable, name));

  if (!opFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&symbolTable->getRegion(0).front());

    auto opFunctionTy = FunctionType::get(context, operandTypes, resultTypes);
    opFunc = rewriter.create<func::FuncOp>(
        rewriter.getUnknownLoc(), name, opFunctionTy);

    opFunc.setPrivate();
    opFunc->setAttr(
        LLVM::LLVMDialect::getReadnoneAttrName(), UnitAttr::get(context));
  }
  assert(
      isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(symbolTable, name)));

  rewriter.replaceOpWithNewOp<func::CallOp>(
      op, name, resultTypes, op->getOperands());
  return success();
}

template <typename OpTy>
void populatePatternsForOp(RewritePatternSet &patterns, MLIRContext *ctx,
    StringRef opString, uint8_t n_bits, uint8_t es_val) {
  patterns.add<ArithToPositFuncLowering<OpTy>>(ctx, opString, n_bits, es_val);
}

void mlir::populateArithToPositFuncConversionPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  populatePatternsForOp<arith::AddFOp>(patterns, ctx, "add", 8, 0);
  populatePatternsForOp<arith::MulFOp>(patterns, ctx, "mul", 16, 1);
}

std::unique_ptr<mlir::Pass> mlir::createConvertArithToPositFuncPass() {
  return std::make_unique<ConvertArithToPositFuncPass>();
}
