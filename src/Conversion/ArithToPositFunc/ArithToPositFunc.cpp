#include "src/Conversion/ArithToPositFunc/ArithToPositFunc.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "convert-arith-to-posit-func"

using namespace mlir;

struct FloatToUIntTypeConverter : public mlir::TypeConverter {
  FloatToUIntTypeConverter() {
    addConversion([](Type type) -> Type {
      if (isa<Float32Type>(type)) {
        return IntegerType::get(type.getContext(), 32, IntegerType::Signless);
      }
      return type;
    });
  }
};

struct ConvertArithToPositFuncPass
    : public PassWrapper<ConvertArithToPositFuncPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
  StringRef getArgument() const override;
  StringRef getDescription() const override;
};

bool isUint32Type(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth() == 32 && intType.isSignless();
  }
  return false;
}

void ConvertArithToPositFuncPass::runOnOperation() {
  auto module = getOperation();
  RewritePatternSet patterns(&getContext());

  FloatToUIntTypeConverter typeConverter;

  // mlir::populateArithToPositFuncConversionPatterns(patterns);
  populateConvertArithConstantFloatToUIntPattern(patterns, typeConverter);

  ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect>();
  // target.addLegalDialect<arith::ArithDialect>();
  // target.addIllegalDialect<arith::ArithDialect>();
  target.addDynamicallyLegalOp<arith::ConstantOp>(
      [&](arith::ConstantOp op) { return isUint32Type(op.getType()); });
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

StringRef ConvertArithToPositFuncPass::getArgument() const {
  return "convert-arith-to-posit-func";
}

StringRef ConvertArithToPositFuncPass::getDescription() const {
  return "Lower the arith dialect to posit func dialect.";
}

struct ConvertArithConstantFloatToUIntPattern
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op,
      typename arith::ConstantOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // support only float32
    if (!isa<Float32Type>(op.getType()))
      return failure();

    // get the float value
    auto floatAttr = dyn_cast<FloatAttr>(op.getValue());
    if (!floatAttr)
      return failure();

    APFloat apFloat = floatAttr.getValue();
    uint64_t floatBits = apFloat.bitcastToAPInt().getZExtValue();
    uint32_t uintValue = static_cast<uint32_t>(floatBits >> 1);

    auto UIntType = getTypeConverter()->convertType(op.getType());

    if (!UIntType)
      return failure();

    // auto UIntAttr = rewriter.getUI32IntegerAttr(uintValue);
    auto UIntAttr = rewriter.getIntegerAttr(UIntType, uintValue);
    auto newOp =
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, UIntType, UIntAttr);
    llvm::errs() << "After replaceOpWithNewOp:\n";
    if (newOp) {
      newOp.print(llvm::errs());
    } else {
      llvm::errs() << "Operation was not replaced\n";
    }
    // llvm::outs() << "Float value: uwu" << uintValue << "\n";
    return success();
  }
};

void mlir::populateConvertArithConstantFloatToUIntPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<ConvertArithConstantFloatToUIntPattern>(typeConverter, ctx);
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

void mlir::populateArithToPositFuncConversionPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  populatePatternsForOp<arith::AddFOp>(patterns, ctx, "add", 8, 0);
  populatePatternsForOp<arith::MulFOp>(patterns, ctx, "mul", 16, 1);
}

std::unique_ptr<mlir::Pass> mlir::createConvertArithToPositFuncPass() {
  return std::make_unique<ConvertArithToPositFuncPass>();
}
