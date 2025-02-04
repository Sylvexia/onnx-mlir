#include "src/Conversion/ArithToPositFunc/ArithToPositFunc.hpp"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include <set>

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

  // llvm::errs() << "scale: " << (int)scale << "\n";
  // llvm::errs() << "es_val: " << (int)es_val << "\n";
  // llvm::errs() << "regime: " << regime << "\n";
  // llvm::errs() << "regime len: " << regime_len << "\n";

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

  // llvm::errs() << "exp_frac: " << exp_frac << "\n";

  int exp_frac_len = 0;
  if (fraction_len == 0)
    exp_frac_len = es_val - countTrailZero(exponent);
  else
    exp_frac_len = es_val + fraction_len;

  int diff_bit_len = abs(exp_frac_len - trailing_len);
  if (exp_frac_len > trailing_len) {
    // the rounding scheme is to be verified
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
  // for (int i = n_bits - 1; i >= 0; i--) {
  //   llvm::errs() << ((result >> i) & 1);
  // }
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
    addConversion([](Type type) -> Type { return type; });
    addConversion([bitWidth](MemRefType type) -> Type {
      if (type.getElementType().isF32())
        return type.clone(
            type.getShape(), IntegerType::get(type.getContext(), bitWidth,
                                 IntegerType::Signless));
      return type;
    });
    addConversion([bitWidth](TensorType type) -> Type {
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

template <typename Op>
struct MemrefAllocationToIntPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  MemrefAllocationToIntPattern(
      const TypeConverter &typeConverter, MLIRContext *context)
      : mlir::OpConversionPattern<Op>(typeConverter, context){};

  LogicalResult matchAndRewrite(Op op, typename Op::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto memRefType = cast<MemRefType>(op.getType());

    if (!isa<Float32Type>(memRefType.getElementType()))
      return failure();

    // why do we need "this"?
    auto newMemRefType =
        cast<MemRefType>(this->getTypeConverter()->convertType(memRefType));

    if (!newMemRefType)
      return failure();

    rewriter.replaceOpWithNewOp<Op>(op, newMemRefType, op.getDynamicSizes(),
        op.getSymbolOperands(), op.getAlignmentAttr());

    return success();
  }
};

template <typename... Ops>
void populateMemrefAllocationToIntPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();
  (patterns.add<MemrefAllocationToIntPattern<Ops>>(typeConverter, ctx), ...);
}

struct MemrefAllocToIntPattern : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  MemrefAllocToIntPattern(
      const TypeConverter &typeConverter, MLIRContext *context)
      : mlir::OpConversionPattern<memref::AllocOp>(typeConverter, context){};

  LogicalResult matchAndRewrite(memref::AllocOp op,
      typename memref::AllocOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto memRefType = cast<MemRefType>(op.getType());

    auto newMemRefResType =
        cast<MemRefType>(getTypeConverter()->convertType(memRefType));

    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, newMemRefResType,
        op.getDynamicSizes(), adaptor.getSymbolOperands(),
        op.getAlignmentAttr());

    return success();
  }
};

struct MemRefStoreOpToIntPattern : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  MemRefStoreOpToIntPattern(
      const TypeConverter &typeConverter, MLIRContext *context)
      : mlir::OpConversionPattern<memref::StoreOp>(typeConverter, context){};

  LogicalResult matchAndRewrite(memref::StoreOp op,
      typename memref::StoreOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto memRefOperand = op.getMemRef();
    auto memRefType = dyn_cast<MemRefType>(memRefOperand.getType());

    if (!memRefType)
      return failure();

    if (!isa<Float32Type>(memRefType.getElementType()))
      return failure();

    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, adaptor.getValue(), adaptor.getMemref(), op.getIndices());

    return success();
  }
};

struct MemRefLoadOpToIntPattern : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  MemRefLoadOpToIntPattern(
      const TypeConverter &typeConverter, MLIRContext *context)
      : mlir::OpConversionPattern<memref::LoadOp>(typeConverter, context){};

  LogicalResult matchAndRewrite(memref::LoadOp op,
      typename memref::LoadOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto memRefOperand = op.getMemRef();
    auto memRefType = dyn_cast<MemRefType>(memRefOperand.getType());

    if (!memRefType)
      return failure();

    if (!isa<Float32Type>(memRefType.getElementType()))
      return failure();

    rewriter.replaceOpWithNewOp<memref::LoadOp>(
        op, adaptor.getMemref(), op.getIndices(), op.getNontemporalAttr());

    return success();
  }
};

void populateMemRefLoadOpToIntPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<MemRefLoadOpToIntPattern>(typeConverter, ctx);
}

struct MemRefReinterpretCastOpToIntPattern
    : public OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern<memref::ReinterpretCastOp>::OpConversionPattern;

  MemRefReinterpretCastOpToIntPattern(
      const TypeConverter &typeConverter, MLIRContext *context)
      : mlir::OpConversionPattern<memref::ReinterpretCastOp>(
            typeConverter, context){};

  LogicalResult matchAndRewrite(memref::ReinterpretCastOp op,
      typename memref::ReinterpretCastOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Value newSource = adaptor.getSource();
    auto newResultType = dyn_cast<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (!newResultType)
      return failure();

    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(op, newResultType,
        newSource, op.getMixedOffsets()[0], op.getMixedSizes(),
        op.getMixedStrides(), op->getAttrs());

    return success();
  }
};

void populateReinterpretCastOpToIntPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<MemRefReinterpretCastOpToIntPattern>(typeConverter, ctx);
}

struct MemrefDimOpToIntPattern : public OpConversionPattern<memref::DimOp> {
  using OpConversionPattern<memref::DimOp>::OpConversionPattern;

  MemrefDimOpToIntPattern(
      const TypeConverter &typeConverter, MLIRContext *context)
      : mlir::OpConversionPattern<memref::DimOp>(typeConverter, context){};

  LogicalResult matchAndRewrite(memref::DimOp op,
      typename memref::DimOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<memref::DimOp>(
        op, adaptor.getSource(), op.getIndex());

    return success();
  }
};

struct KrnlMemcpyOpToIntPattern : public OpConversionPattern<KrnlMemcpyOp> {
  using OpConversionPattern<KrnlMemcpyOp>::OpConversionPattern;

  KrnlMemcpyOpToIntPattern(
      const TypeConverter &typeConverter, MLIRContext *context)
      : mlir::OpConversionPattern<KrnlMemcpyOp>(typeConverter, context){};

  LogicalResult matchAndRewrite(KrnlMemcpyOp op,
      typename KrnlMemcpyOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto destElementTy =
        cast<MemRefType>(op.getDest().getType()).getElementType();
    auto srcElementTy =
        cast<MemRefType>(op.getSrc().getType()).getElementType();

    if (!isa<Float32Type>(destElementTy) || !isa<Float32Type>(srcElementTy))
      return failure();

    rewriter.replaceOpWithNewOp<KrnlMemcpyOp>(op, adaptor.getDest(),
        adaptor.getSrc(), op.getNumElems(), op.getDestOffset(),
        op.getSrcOffset());

    return success();
  }
};

struct KrnlGlobalOpToIntPattern : public OpConversionPattern<KrnlGlobalOp> {
  using OpConversionPattern<KrnlGlobalOp>::OpConversionPattern;

  KrnlGlobalOpToIntPattern(const TypeConverter &typeConverter,
      MLIRContext *context, uint8_t n_bits, uint8_t es_val)
      : mlir::OpConversionPattern<KrnlGlobalOp>(typeConverter, context),
        n_bits(n_bits), es_val(es_val){};

  LogicalResult matchAndRewrite(KrnlGlobalOp op,
      typename KrnlGlobalOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto type = op.getType();
    auto memRefType = cast<MemRefType>(type);
    if (!isa<Float32Type>(memRefType.getElementType()))
      return failure();

    auto newMemRefType = getTypeConverter()->convertType(memRefType);

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

    rewriter.replaceOpWithNewOp<KrnlGlobalOp>(op, newMemRefType, op.getShape(),
        op.getNameAttr(), newDenseAttr, op.getOffsetAttr(),
        op.getAlignmentAttr());

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
    // llvm::errs() << "float value: " << apFloat.convertToFloat() << "\n";

    auto IntType = getTypeConverter()->convertType(op.getType());
    auto uintValue = convertFloat32ToPosit(floatBits, n_bits, es_val);

    if (!IntType)
      return failure();

    auto IntAttr = rewriter.getIntegerAttr(IntType, uintValue);
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

template <typename Op> // TODO: this is not binop but returntype int
struct ConvertArithBinOpToPositFuncLowering : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

public:
  ConvertArithBinOpToPositFuncLowering(const TypeConverter &typeConverter,
      MLIRContext *context, StringRef opString, uint8_t n_bits, uint8_t es_val)
      : mlir::OpConversionPattern<Op>(typeConverter, context),
        opString(opString), n_bits(n_bits), es_val(es_val){};

  LogicalResult matchAndRewrite(Op op, typename Op::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {

    // this only support scalar, return failure if its vector
    if (isa<VectorType>(op->getResult(0).getType()))
      return failure();

    if (!isa<Float32Type>(op.getType()))
      return failure();

    std::string name = getPositFuncStr(n_bits, es_val, opString);

    auto returnType =
        this->getTypeConverter()->convertType(op->getOpResult(0).getType());

    if (!returnType)
      return failure();

    auto module = SymbolTable::getNearestSymbolTable(op);
    auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(module, name));
    if (!opFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&module->getRegion(0).front());

      auto opFunctionTy = FunctionType::get(
          rewriter.getContext(), adaptor.getOperands().getTypes(), returnType);
      opFunc = rewriter.create<func::FuncOp>(
          rewriter.getUnknownLoc(), name, opFunctionTy);

      opFunc.setPrivate();
      opFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(),
          UnitAttr::get(rewriter.getContext()));
    }
    assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module, name)));

    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, name, returnType, adaptor.getOperands());

    return success();
  }

private:
  std::string opString;
  uint8_t n_bits;
  uint8_t es_val;
};

template <typename OpType>
void populateOpSigToPositPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, const std::string &opName, int nBits,
    int esVal) {
  patterns.add<ConvertArithBinOpToPositFuncLowering<OpType>>(
      typeConverter, patterns.getContext(), opName, nBits, esVal);
}

// chance to merge with template?
struct ConvertArithCmpToPositFuncLowering
    : public OpConversionPattern<arith::CmpFOp> {
  using OpConversionPattern<arith::CmpFOp>::OpConversionPattern;

public:
  ConvertArithCmpToPositFuncLowering(const TypeConverter &typeConverter,
      MLIRContext *context, uint8_t n_bits, uint8_t es_val)
      : mlir::OpConversionPattern<arith::CmpFOp>(typeConverter, context),
        n_bits(n_bits), es_val(es_val),
        get_pred_str{{arith::CmpFPredicate::OEQ, "oeq"},
            {arith::CmpFPredicate::OGT, "ogt"},
            {arith::CmpFPredicate::OLT, "olt"},
            {arith::CmpFPredicate::OGE, "oge"},
            {arith::CmpFPredicate::OLE, "ole"},
            {arith::CmpFPredicate::UNE, "une"},
            {arith::CmpFPredicate::ORD, "ord"},
            {arith::CmpFPredicate::UNO, "uno"},
            {arith::CmpFPredicate::UEQ, "ueq"},
            {arith::CmpFPredicate::UGT, "ugt"},
            {arith::CmpFPredicate::ULT, "ult"},
            {arith::CmpFPredicate::UGE, "uge"},
            {arith::CmpFPredicate::ULE, "ule"},
            {arith::CmpFPredicate::AlwaysTrue, "true"},
            {arith::CmpFPredicate::AlwaysFalse, "false"}} {};

  LogicalResult matchAndRewrite(arith::CmpFOp op,
      typename arith::CmpFOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {

    // this only support scalar, return failure if its vector
    if (isa<VectorType>(op->getResult(0).getType()))
      return failure();

    if (!isa<Float32Type>(op->getOperand(0).getType()))
      return failure();

    if (!isa<Float32Type>(op->getOperand(1).getType()))
      return failure();

    auto predID = (op.getPredicateAttr().getValue());
    std::string name = getPositFuncStr(n_bits, es_val, get_pred_str.at(predID));

    auto returnType = rewriter.getI1Type();

    if (!returnType)
      return failure();

    auto module = SymbolTable::getNearestSymbolTable(op);
    auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(module, name));
    if (!opFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&module->getRegion(0).front());

      auto opFunctionTy = FunctionType::get(
          rewriter.getContext(), adaptor.getOperands().getTypes(), returnType);
      opFunc = rewriter.create<func::FuncOp>(
          rewriter.getUnknownLoc(), name, opFunctionTy);

      opFunc.setPrivate();
      opFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(),
          UnitAttr::get(rewriter.getContext()));
    }
    assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module, name)));

    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, name, returnType, adaptor.getOperands());

    return success();
  }

private:
  uint8_t n_bits;
  uint8_t es_val;
  std::unordered_map<arith::CmpFPredicate, std::string> get_pred_str;
};

struct ConvertArithSitofpToPositFuncLowering
    : public OpConversionPattern<arith::SIToFPOp> {
  using OpConversionPattern<arith::SIToFPOp>::OpConversionPattern;

public:
  ConvertArithSitofpToPositFuncLowering(const TypeConverter &typeConverter,
      MLIRContext *context, uint8_t n_bits, uint8_t es_val)
      : mlir::OpConversionPattern<arith::SIToFPOp>(typeConverter, context),
        n_bits(n_bits), es_val(es_val){};

  LogicalResult matchAndRewrite(arith::SIToFPOp op, arith::SIToFPOp::Adaptor,
      ConversionPatternRewriter &rewriter) const final {

    // this only support scalar, return failure if its vector
    if (isa<VectorType>(op->getResult(0).getType()))
      return failure();

    if (!isa<IntegerType>(op.getOperand().getType()))
      return failure();

    auto name = getPositFuncStr(n_bits, es_val, "sitofp");

    auto returnType =
        getTypeConverter()->convertType(op->getOpResult(0).getType());

    if (!returnType)
      return failure();

    auto module = SymbolTable::getNearestSymbolTable(op);
    auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(module, name));
    if (!opFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&module->getRegion(0).front());

      // operand is original, so adaptor is not needed
      auto opFunctionTy = FunctionType::get(
          rewriter.getContext(), op.getOperand().getType(), returnType);
      opFunc = rewriter.create<func::FuncOp>(
          rewriter.getUnknownLoc(), name, opFunctionTy);

      opFunc.setPrivate();
      opFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(),
          UnitAttr::get(rewriter.getContext()));
    }
    assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module, name)));

    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, name, returnType, op.getOperand());

    return success();
  }

private:
  uint8_t n_bits;
  uint8_t es_val;
};

struct ConvertArithFptosiToPositFuncLowering
    : public OpConversionPattern<arith::FPToSIOp> {
  using OpConversionPattern<arith::FPToSIOp>::OpConversionPattern;

public:
  ConvertArithFptosiToPositFuncLowering(const TypeConverter &typeConverter,
      MLIRContext *context, uint8_t n_bits, uint8_t es_val)
      : mlir::OpConversionPattern<arith::FPToSIOp>(typeConverter, context),
        n_bits(n_bits), es_val(es_val){};

  LogicalResult matchAndRewrite(arith::FPToSIOp op,
      arith::FPToSIOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    auto resultType = op.getResult().getType();

    if (!isa<IntegerType>(resultType))
      return failure();

    auto name = getPositFuncStr(n_bits, es_val, "fptosi");
    auto operandType = op->getOperands();

    bool allFP32 = llvm::all_of(operandType, [](mlir::Value operand) {
      return isa<Float32Type>(operand.getType());
    });
    
    if (!allFP32)
      return failure();

    auto module = SymbolTable::getNearestSymbolTable(op);
    auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(module, name));
    if (!opFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&module->getRegion(0).front());

      auto opFunctionTy = FunctionType::get(
          rewriter.getContext(), adaptor.getOperands(), resultType);
      opFunc = rewriter.create<func::FuncOp>(
          rewriter.getUnknownLoc(), name, opFunctionTy);

      opFunc.setPrivate();
      opFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(),
          UnitAttr::get(rewriter.getContext()));
    }
    assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module, name)));

    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, name, resultType, adaptor.getOperands());

    return success();
  }

private:
  uint8_t n_bits;
  uint8_t es_val;
};

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
  Option<int> _es_val{
      *this, "es-val", llvm::cl::desc("Number of es value"), llvm::cl::init(0)};
};

void ConvertArithToPositFuncPass::runOnOperation() {
  {
    auto module = getOperation();
    std::set<std::string> operationNames;
    module.walk([&](Operation *op) {
      operationNames.insert(op->getName().getStringRef().str());
    });

    for (const auto &name : operationNames) {
      llvm::errs() << "Saw operation: " << name << "\n";
    }
  }
  auto module = getOperation();

  // vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  // vector::populateVectorBroadcastLoweringPatterns(patterns);
  // vector::populateVectorContractLoweringPatterns(
  //     patterns, vector::VectorTransformsOptions());
  // vector::populateVectorTransposeLoweringPatterns(
  //     patterns, vector::VectorTransformsOptions());

  // RewritePatternSet prePatterns(&getContext());

  RewritePatternSet patterns(&getContext());
  FloatToIntTypeConverter typeConverter(_n_bits);

  auto mapOpTypeToOpstring = [&](auto opType, const std::string &opString) {
    populateOpSigToPositPattern<decltype(opType)>(
        patterns, typeConverter, opString, _n_bits, _es_val);
  };

  mapOpTypeToOpstring(arith::AddFOp{}, "add");
  mapOpTypeToOpstring(arith::SubFOp{}, "sub");
  mapOpTypeToOpstring(arith::MulFOp{}, "mul");
  mapOpTypeToOpstring(arith::DivFOp{}, "div");
  mapOpTypeToOpstring(arith::NegFOp{}, "neg");
  mapOpTypeToOpstring(arith::MaxNumFOp{}, "maxnum");
  mapOpTypeToOpstring(arith::MinNumFOp{}, "minnum");
  mapOpTypeToOpstring(arith::SelectOp{}, "select");
  mapOpTypeToOpstring(arith::MaximumFOp{}, "max");
  mapOpTypeToOpstring(arith::MinimumFOp{}, "min");
  mapOpTypeToOpstring(math::AbsFOp{}, "abs");
  mapOpTypeToOpstring(math::SqrtOp{}, "sqrt");
  mapOpTypeToOpstring(math::RsqrtOp{}, "rsqrt");
  mapOpTypeToOpstring(math::ExpOp{}, "exp");
  mapOpTypeToOpstring(math::SinOp{}, "sin");
  mapOpTypeToOpstring(math::CosOp{}, "cos");
  mapOpTypeToOpstring(math::TanOp{}, "tan");
  mapOpTypeToOpstring(math::AsinOp{}, "asin");
  mapOpTypeToOpstring(math::AcosOp{}, "acos");
  mapOpTypeToOpstring(math::AtanOp{}, "atan");
  mapOpTypeToOpstring(math::SinhOp{}, "sinh");
  mapOpTypeToOpstring(math::CoshOp{}, "cosh");
  mapOpTypeToOpstring(math::TanhOp{}, "tanh");
  mapOpTypeToOpstring(math::ErfOp{}, "erf");
  mapOpTypeToOpstring(math::LogOp{}, "log");
  mapOpTypeToOpstring(math::FloorOp{}, "floor");
  mapOpTypeToOpstring(math::CeilOp{}, "ceil");
  mapOpTypeToOpstring(math::TruncOp{}, "trunc");
  mapOpTypeToOpstring(math::RoundOp{}, "round");

  patterns.add<ConvertArithCmpToPositFuncLowering>(
      typeConverter, patterns.getContext(), _n_bits, _es_val);
  patterns.add<ConvertArithSitofpToPositFuncLowering>(
      typeConverter, patterns.getContext(), _n_bits, _es_val);
  patterns.add<ConvertArithFptosiToPositFuncLowering>(
      typeConverter, patterns.getContext(), _n_bits, _es_val);

  populateConvertArithConstantFloatToIntPattern(
      patterns, typeConverter, _n_bits, _es_val);

  populateKrnlGlobalOpToIntPattern(patterns, typeConverter, _n_bits, _es_val);
  patterns.add<KrnlMemcpyOpToIntPattern>(typeConverter, patterns.getContext());
  patterns.add<MemrefDimOpToIntPattern>(typeConverter, patterns.getContext());

  populateMemrefAllocationToIntPattern<memref::AllocaOp, memref::AllocOp>(
      patterns, typeConverter); // getType() same builder pattern
  populateMemRefLoadOpToIntPattern(patterns, typeConverter);
  patterns.add<MemRefStoreOpToIntPattern>(typeConverter, patterns.getContext());
  populateReinterpretCastOpToIntPattern(patterns, typeConverter);

  // populate standard lowering
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);

  ConversionTarget target(getContext());
  target.addDynamicallyLegalOp<arith::ConstantOp, arith::AddFOp, arith::SubFOp,
      arith::MulFOp, arith::DivFOp, arith::NegFOp, arith::MaxNumFOp,
      arith::MinNumFOp, arith::CmpFOp, arith::SelectOp, arith::MaximumFOp,
      arith::MinimumFOp, math::AbsFOp, math::SqrtOp, math::RsqrtOp, math::ExpOp,
      math::SinOp, math::CosOp, math::TanOp, math::AsinOp, math::AcosOp,
      math::AtanOp, math::SinhOp, math::CoshOp, math::TanhOp, math::LogOp,
      math::FloorOp, math::CeilOp, math::TruncOp, math::RoundOp, math::ErfOp,
      arith::SIToFPOp, arith::FPToSIOp>(
      [&](Operation *op) { return typeConverter.isLegal(op); });

  target.addDynamicallyLegalOp<KrnlGlobalOp>([&](KrnlGlobalOp op) {
    return typeConverter.isLegal(
        cast<MemRefType>(op->getResult(0).getType()).getElementType());
  });

  target.addDynamicallyLegalOp<KrnlMemcpyOp>([&](KrnlMemcpyOp op) {
    auto destElementTy = cast<MemRefType>(op.getDest().getType());
    auto srcElementTy = cast<MemRefType>(op.getSrc().getType());
    return typeConverter.isLegal(destElementTy) &&
           typeConverter.isLegal(srcElementTy);
  });

  target.addDynamicallyLegalOp<memref::DimOp>([&](memref::DimOp op) {
    auto sourceType = cast<MemRefType>(op.getSource().getType());
    return typeConverter.isLegal(sourceType);
  });

  target.addDynamicallyLegalOp<memref::AllocaOp, memref::AllocOp>(
      [&](Operation *op) {
        return typeConverter.isLegal(
            cast<MemRefType>(op->getResult(0).getType()));
        // return typeConverter.isLegal(op.getType().getElementType());
      });

  target.addDynamicallyLegalOp<memref::LoadOp, memref::ReinterpretCastOp>(
      [&](Operation *op) { return typeConverter.isLegal(op); });

  target.addDynamicallyLegalOp<memref::StoreOp>([&](memref::StoreOp op) {
    return typeConverter.isLegal(cast<MemRefType>(op.getMemref().getType()));
  });

  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    bool res = typeConverter.isSignatureLegal(op.getFunctionType()) &&
               typeConverter.isLegal(&op.getBody());
    return res;
  });

  target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
    bool res = typeConverter.isLegal(op);
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

  scf::populateSCFStructuralTypeConversionsAndLegality(
      typeConverter, patterns, target);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::createConvertArithToPositFuncPass() {
  return std::make_unique<ConvertArithToPositFuncPass>();
}

std::unique_ptr<mlir::Pass> mlir::createConvertArithToPositFuncPass(
    uint8_t n_bits, uint8_t es_val) {
  return std::make_unique<ConvertArithToPositFuncPass>(n_bits, es_val);
}

struct CustomLowerAffinePass
    : public PassWrapper<CustomLowerAffinePass, OperationPass<ModuleOp>> {
  void runOnOperation() final {
    auto module = getOperation();
    RewritePatternSet patterns(&getContext());
    populateAffineToStdConversionPatterns(patterns);

    ConversionTarget target(getContext());
    target.addIllegalDialect<mlir::affine::AffineDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
  StringRef getArgument() const override { return "convert-affine-to-cf-func"; }
  StringRef getDescription() const override {
    return "Lower the affine dialect to cf dialect.";
  }
};

std::unique_ptr<mlir::Pass> mlir::createCustomLowerAffinePass() {
  return std::make_unique<CustomLowerAffinePass>();
}