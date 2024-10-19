#include "src/Conversion/ArithToPositFunc/ArithToPositFunc.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"

#define DEBUG_TYPE "convert-arith-to-posit-func"

using namespace mlir;

int countTrailZero(uint64_t value) {
  if (value == 0)
    return 0;
  return __builtin_ctzl(value);
}

uint64_t removeTrailZero(uint64_t value) {
  if (value == 0)
    return 0;
  return value >> countTrailZero(value);
}

int bit_length(uint64_t value) {
  return std::numeric_limits<uint64_t>::digits - __builtin_clzl(value);
}

uint64_t convertFloat32ToPosit(
    uint64_t raw_bit, uint8_t n_bits, uint8_t es_val) {
  // uint64_t raw_bit = *reinterpret_cast<uint64_t *>(&value);

  uint64_t result = 0;

  uint8_t n_raw = 32;
  uint8_t n_frac = 23;
  uint8_t n_exp = 8;
  uint bias = 127;

  if ((raw_bit & ((1ULL << (n_raw - 1)) - 1)) == 0) {
    result = 0;
    return result;
  }

  if ((raw_bit & ((1ULL << (n_raw - 1)) - 1)) >=
      (((1ULL << n_exp) - 1) << n_frac)) {
    result = (1ULL << n_bits) - 1;
    return result;
  }

  bool sign = (raw_bit >> (n_raw - 1)) & 1;
  int scale = ((raw_bit & ((1ULL << (n_raw - 1)) - 1)) >> n_frac) - bias;
  uint64_t fraction = (1ULL << n_frac) | (raw_bit & ((1ULL << n_frac) - 1));

  int regime = scale >> es_val;
  int regime_len = (regime >= 0) ? regime + 2 : -regime + 1;

  llvm::errs() << "scale: " << (int)scale << "\n";
  llvm::errs() << "es_val: " << (int)es_val << "\n";
  llvm::errs() << "regime: " << regime << "\n";
  llvm::errs() << "regime len: " << regime_len << "\n";

  // this should be long long int for 64-bit
  long long int exponent = scale & ((1ULL << es_val) - 1);

  // check if regime is out of range
  if (regime_len >= n_bits + 1) {
    if (regime >= 0)
      result = (1ULL << (n_bits - 1)) - 1; // max posit
    else
      result = 1; // min posit

    if (sign)
      result = (1 << (n_bits - 1)) | result;
    return result;
  }

  // encode regime
  result = 0;
  if (regime >= 0)
    result |= (((1ULL << (regime_len - 1)) - 1) << (n_bits - regime_len));
  else if (n_bits - 1 >= regime_len)
    result |= ((1ULL << (n_bits - 1 - regime_len)));

  fraction = removeTrailZero(fraction);
  int fraction_len = bit_length(fraction) - 1;
  fraction &= ((1ULL << fraction_len) - 1);
  int trailing_len = n_bits - regime_len - 1;
  uint64_t exp_frac = removeTrailZero((exponent << fraction_len) | fraction);

  llvm::errs() << "exp_frac: " << exp_frac << "\n";

  int exp_frac_len = 0;
  if (fraction_len == 0)
    exp_frac_len = es_val - countTrailZero(exponent);
  else
    exp_frac_len = es_val + fraction_len;

  int diff_bit_len = abs(exp_frac_len - trailing_len);
  if (exp_frac_len > trailing_len) {
    // this might be wrong
    bool guard, round, sticky;
    guard = (exp_frac >> (diff_bit_len - 1)) & 1;
    round = (exp_frac >> (diff_bit_len - 2)) & 1;
    sticky = (exp_frac & ((1ULL << (diff_bit_len - 2)) - 1));
    bool round_up = guard & (round | sticky);
    result |= (exp_frac >> diff_bit_len);
    if (round_up)
      result += 1;
  } else {
    result |= exp_frac << (diff_bit_len);
  }

  if (sign)
    result |= 1 << (n_bits - 1);

  // log result as binary
  for (int i = n_bits - 1; i >= 0; i--) {
    llvm::errs() << ((result >> i) & 1);
  }
  return result;
}

// e.g. posit8es0_add
std::string getPositFuncStr(
    uint8_t n_bits, uint8_t es_val, std::string opString) {
  return "posit" + std::to_string(n_bits) + "es" + std::to_string(es_val) +
         "_" + opString;
}

struct FloatToIntTypeConverter : public mlir::TypeConverter {
  explicit FloatToIntTypeConverter(uint8_t bitWidth) {
    addConversion([](Type type) -> Type {
      return type;
    });
    addConversion([bitWidth](MemRefType type) -> Type {
      llvm::errs() << "memref type: " << type << "\n";
      if (type.getElementType().isF32())
        return MemRefType::get(
            type.getShape(), IntegerType::get(type.getContext(), bitWidth,
                                 IntegerType::Signless));
      return type;
    });
    addConversion([bitWidth](TensorType type) -> Type {
      llvm::errs() << "tensor type: " << type << "\n";
      if (type.getElementType().isF32())
        return type.clone(
            type.getShape(), IntegerType::get(type.getContext(), bitWidth,
                                 IntegerType::Signless));
      return type;
    });
    addConversion([bitWidth](FloatType type) -> Type {
      if (isa<Float32Type>(type)) {
        return IntegerType::get(
            type.getContext(), bitWidth, IntegerType::Signless);
      }
      return type;
    });
  }
};

bool isIntType(Type type, uint8_t bitWidth) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth() == bitWidth && intType.isSignless();
  }
  return false;
}

struct ConvertArithToPositFuncPass
    : public PassWrapper<ConvertArithToPositFuncPass, OperationPass<ModuleOp>> {
  void runOnOperation() final;
  StringRef getArgument() const override {
    return "convert-arith-to-posit-func";
  }
  StringRef getDescription() const override {
    return "Lower the arith dialect to posit func dialect.";
  };

  ConvertArithToPositFuncPass() = default;
  ConvertArithToPositFuncPass(const ConvertArithToPositFuncPass &pass)
      : PassWrapper<ConvertArithToPositFuncPass, OperationPass<ModuleOp>>() {}
  ConvertArithToPositFuncPass(uint8_t n_bits, uint8_t es_val) {
    this->_n_bits = n_bits;
    this->_es_val = es_val;
  }

public:
  Option<int> _n_bits{*this, "n-bits",
      llvm::cl::desc("Number of bits in posit"), llvm::cl::init(8)};
  Option<int> _es_val{*this, "es-val",
      llvm::cl::desc("Number of bits in exponent"), llvm::cl::init(0)};
};

void ConvertArithToPositFuncPass::runOnOperation() {
  auto module = getOperation();
  RewritePatternSet patterns(&getContext());

  FloatToIntTypeConverter typeConverter(_n_bits);

  // custom lowering
  populateConvertArithAddToPositFuncPattern(
      patterns, typeConverter, "add", _n_bits, _es_val);
  populateConvertArithConstantFloatToIntPattern(
      patterns, typeConverter, _n_bits, _es_val);
  populateKrnlGlobalOpToIntPattern(patterns, typeConverter, _n_bits, _es_val);

  // populate standard lowering
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);

  ConversionTarget target(getContext());
  target.addIllegalDialect<arith::ArithDialect>();
  target.addDynamicallyLegalOp<arith::ConstantOp>(
      [&](arith::ConstantOp op) { return isIntType(op.getType(), _n_bits); });

  target.addDynamicallyLegalOp<KrnlGlobalOp>([&](KrnlGlobalOp op) {
    return isIntType(
        cast<MemRefType>(op->getResult(0).getType()).getElementType(), _n_bits);
  });

  // target.addDynamicallyLegalDialect<memref::MemRefDialect>(
  //     [&typeConverter](Operation *op) { return typeConverter.isLegal(op); });

  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    bool res = typeConverter.isSignatureLegal(op.getFunctionType()) &&
               typeConverter.isLegal(&op.getBody());
    llvm::errs() << "func op: " << res << "\n";
    return res;
  });

  target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
    bool res = typeConverter.isLegal(op);
    llvm::errs() << "return op: " << res << "\n";
    return res;
  });

  target.addDynamicallyLegalOp<func::CallOp>(
      [&](func::CallOp op) { return typeConverter.isLegal(op); });

  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
           isLegalForBranchOpInterfaceTypeConversionPattern(
               op, typeConverter) ||
           isLegalForReturnOpTypeConversionPattern(op, typeConverter);
  });

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

struct KrnlGlobalOpToIntPattern : public OpConversionPattern<KrnlGlobalOp> {
  using OpConversionPattern<KrnlGlobalOp>::OpConversionPattern;

  KrnlGlobalOpToIntPattern(const TypeConverter &typeConverter,
      MLIRContext *context, uint8_t n_bits, uint8_t es_val)
      : mlir::OpConversionPattern<KrnlGlobalOp>(typeConverter, context),
        n_bits(n_bits), es_val(es_val){};

  LogicalResult matchAndRewrite(KrnlGlobalOp op,
      typename KrnlGlobalOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto type = op->getResult(0).getType();
    auto memRefType = cast<MemRefType>(type);
    auto elementType = memRefType.getElementType();

    if (!isa<Float32Type>(elementType))
      return failure();

    auto newElementType = getTypeConverter()->convertType(elementType);
    if (!newElementType)
      return failure();

    auto newMemRefType = MemRefType::get(memRefType.getShape(), newElementType);

    auto valueAttr = op.getValueAttr();
    // cast to denseElementAttr
    auto denseAttr = dyn_cast<DenseElementsAttr>(valueAttr);
    if (!denseAttr)
      return failure();

    // get element type
    auto denseElementType = denseAttr.getType().getElementType();
    if (!isa<Float32Type>(denseElementType))
      return failure();

    auto newDenseElementType =
        getTypeConverter()->convertType(denseElementType);

    auto newDenseAttr = denseAttr.mapValues(
        newDenseElementType, [&](const APFloat &value) -> APInt {
          uint64_t floatBits = value.bitcastToAPInt().getZExtValue();
          return APInt(newDenseElementType.getIntOrFloatBitWidth(),
              convertFloat32ToPosit(floatBits, n_bits, es_val));
        });

    auto new_op = rewriter.replaceOpWithNewOp<KrnlGlobalOp>(op, newMemRefType,
        op.getShape(), op.getNameAttrName(), newDenseAttr, op.getOffsetAttr(),
        op.getAlignmentAttr());

    llvm::errs() << "new op: " << new_op << "\n";
    llvm::errs() << "uwu" << "\n";

    // log out the original value and the new value

    for (auto [origValue, newValue] : llvm::zip(
             denseAttr.getValues<APFloat>(), newDenseAttr.getValues<APInt>())) {
      llvm::errs() << "orig: " << origValue.convertToFloat() << "\n";
      // cast to binary
      for (int i = n_bits - 1; i >= 0; i--) {
        llvm::errs() << ((newValue.getZExtValue() >> i) & 1);
      }
      llvm::errs() << "\n";
    }

    return success();
  }

private:
  uint8_t n_bits;
  uint8_t es_val;
};

void mlir::populateKrnlGlobalOpToIntPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, uint8_t n_bits, uint8_t es_val) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<KrnlGlobalOpToIntPattern>(typeConverter, ctx, n_bits, es_val);
}

struct ConvertArithConstantFloatToIntPattern
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  ConvertArithConstantFloatToIntPattern(const TypeConverter &typeConverter,
      MLIRContext *context, uint8_t n_bits, uint8_t es_val)
      : mlir::OpConversionPattern<arith::ConstantOp>(typeConverter, context),
        n_bits(n_bits), es_val(es_val){};

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
    llvm::errs() << "float value: " << apFloat.convertToFloat() << "\n";

    auto IntType = getTypeConverter()->convertType(op.getType());
    auto uintValue = convertFloat32ToPosit(floatBits, n_bits, es_val);

    if (!IntType)
      return failure();

    // auto UIntAttr = rewriter.getUI32IntegerAttr(uintValue);
    auto IntAttr = rewriter.getIntegerAttr(IntType, uintValue);
    auto newOp =
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, IntType, IntAttr);
    return success();
  }

private:
  uint8_t n_bits;
  uint8_t es_val;
};

void mlir::populateConvertArithConstantFloatToIntPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, uint8_t n_bits,
    uint8_t es_val) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<ConvertArithConstantFloatToIntPattern>(
      typeConverter, ctx, n_bits, es_val);
}

struct ConvertArithAddToPositFuncLowering
    : public OpConversionPattern<arith::AddFOp> {
  using OpConversionPattern<arith::AddFOp>::OpConversionPattern;

public:
  ConvertArithAddToPositFuncLowering(const TypeConverter &typeConverter,
      MLIRContext *context, StringRef opString, uint8_t n_bits, uint8_t es_val)
      : mlir::OpConversionPattern<arith::AddFOp>(typeConverter, context),
        opString(opString), n_bits(n_bits), es_val(es_val){};

  LogicalResult matchAndRewrite(arith::AddFOp op,
      typename arith::AddFOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {

    // if (!isa<Float32Type>(op.getType()))
    //   return failure();

    std::string name = getPositFuncStr(n_bits, es_val, opString);

    llvm::SmallVector<Type, 2> OperandVec;
    llvm::SmallVector<Type, 1> ResultVec;

    auto operandStatus =
        getTypeConverter()->convertTypes(op->getOperandTypes(), OperandVec);
    auto resultStatus =
        getTypeConverter()->convertTypes(op->getResultTypes(), ResultVec);

    TypeRange operandTypes(OperandVec);
    TypeRange resultTypes(ResultVec);

    auto module = SymbolTable::getNearestSymbolTable(op);
    auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(module, name));
    if (!opFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&module->getRegion(0).front());

      auto opFunctionTy = FunctionType::get(
          rewriter.getContext(), adaptor.getOperands().getTypes(), resultTypes);
      opFunc = rewriter.create<func::FuncOp>(
          rewriter.getUnknownLoc(), name, opFunctionTy);

      opFunc.setPrivate();
      opFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(),
          UnitAttr::get(rewriter.getContext()));
    }
    assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module, name)));

    auto newOp = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, name, resultTypes, adaptor.getOperands());

    return success();
  }

private:
  std::string opString;
  uint8_t n_bits;
  uint8_t es_val;
};

void mlir::populateConvertArithAddToPositFuncPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    StringRef opString, uint8_t n_bits, uint8_t es_val) {
  MLIRContext *context = patterns.getContext();
  patterns.add<ConvertArithAddToPositFuncLowering>(
      typeConverter, context, opString, n_bits, es_val);
}

std::unique_ptr<mlir::Pass> mlir::createConvertArithToPositFuncPass() {
  return std::make_unique<ConvertArithToPositFuncPass>();
}

std::unique_ptr<mlir::Pass> mlir::createConvertArithToPositFuncPass(
    uint8_t n_bits, uint8_t es_val) {
  return std::make_unique<ConvertArithToPositFuncPass>(n_bits, es_val);
}
