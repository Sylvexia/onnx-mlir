#include "src/Conversion/ArithToPositFunc/ArithToPositFunc.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
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
        return MemRefType::get(
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

// todo: can we just use typeConverter.isLegal(op) instead of this?
// bool isIntType(Type type, uint8_t bitWidth) {
//   if (auto intType = dyn_cast<IntegerType>(type)) {
//     return intType.getWidth() == bitWidth && intType.isSignless();
//   }
//   return false;
// }

// bool isIntType(Type type) {
//   if (auto intType = dyn_cast<IntegerType>(type)) {
//     return true;
//   }
//   return false;
// }

// to be renamed to alloc Pattern
template <typename Op>
struct MemRefNoOprandToIntPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  MemRefNoOprandToIntPattern(
      const TypeConverter &typeConverter, MLIRContext *context)
      : mlir::OpConversionPattern<Op>(typeConverter, context){};

  LogicalResult matchAndRewrite(Op op, typename Op::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const final;
};

template <typename Op>
LogicalResult MemRefNoOprandToIntPattern<Op>::matchAndRewrite(Op op,
    typename Op::Adaptor adaptor, ConversionPatternRewriter &rewriter) const {
  auto memRefType = cast<MemRefType>(op.getType());

  if (!isa<Float32Type>(memRefType.getElementType()))
    return failure();

  // why do we need "this"?
  auto newMemRefType =
      cast<MemRefType>(this->getTypeConverter()->convertType(memRefType));

  if (!newMemRefType)
    return failure();

  rewriter.replaceOpWithNewOp<Op>(op, newMemRefType, op.getAlignmentAttr());

  return success();
}

template <typename... Ops>
void populateMemRefNoOprandToIntPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();
  (patterns.add<MemRefNoOprandToIntPattern<Ops>>(typeConverter, ctx), ...);
}

template <typename Op>
struct ReturnTypeToIntPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  ReturnTypeToIntPattern(
      const TypeConverter &typeConverter, MLIRContext *context)
      : mlir::OpConversionPattern<Op>(typeConverter, context){};

  LogicalResult matchAndRewrite(Op op, typename Op::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const final;
};

template <typename Op>
LogicalResult ReturnTypeToIntPattern<Op>::matchAndRewrite(Op op,
    typename Op::Adaptor adaptor, ConversionPatternRewriter &rewriter) const {
  auto memRefType = dyn_cast<MemRefType>(op->getResult(0).getType());
  if (!memRefType)
    return failure();
  if (!isa<Float32Type>(memRefType.getElementType()))
    return failure();

  llvm::errs() << "get f32 type";

  auto newMemRefType =
      dyn_cast<MemRefType>(this->getTypeConverter()->convertType(memRefType));

  llvm::errs() << "converted";

  if (!newMemRefType)
    return failure();

  OperationState newOpState(op->getLoc(), op->getName());
  newOpState.addOperands(adaptor.getOperands());
  newOpState.addTypes(newMemRefType);
  newOpState.addAttributes(op->getAttrs());
  // newOpState.addSuccessors(op->getSucessors());
  auto *newOp = rewriter.create(newOpState);

  llvm::errs() << "new op: " << newOp << "\n";
  rewriter.replaceOp(op, newOp->getResults());
  return success();
}

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

struct AffineForOpToIntPattern
    : public OpConversionPattern<affine::AffineForOp> {
  using OpConversionPattern<affine::AffineForOp>::OpConversionPattern;

  AffineForOpToIntPattern(
      const TypeConverter &typeConverter, MLIRContext *context)
      : mlir::OpConversionPattern<affine::AffineForOp>(
            typeConverter, context){};

  LogicalResult matchAndRewrite(affine::AffineForOp op,
      typename affine::AffineForOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto newForOp =
        rewriter.create<affine::AffineForOp>(loc, op.getLowerBoundOperands(),
            op.getLowerBoundMap(), op.getUpperBoundOperands(),
            op.getUpperBoundMap(), op.getStepAsInt(), adaptor.getInits());

    // the region argument get replaced by the newForOp.getRegion().getArgument
    rewriter.eraseBlock(newForOp.getBody());
    rewriter.inlineRegionBefore(
        adaptor.getRegion(), newForOp.getRegion(), newForOp.getRegion().end());

    auto newIterArgs = newForOp.getRegionIterArgs();
    for (auto &arg : newIterArgs) {
      auto newArgType = getTypeConverter()->convertType(arg.getType());
      if (!newArgType)
        return failure();
      arg.setType(newArgType);
    }

    rewriter.replaceOp(op, newForOp->getResults());

    return success();
  }
};

void populateAffineForOpToIntPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<AffineForOpToIntPattern>(typeConverter, ctx);
}

struct AffineYieldOpToIntPattern
    : public OpConversionPattern<affine::AffineYieldOp> {
  using OpConversionPattern<affine::AffineYieldOp>::OpConversionPattern;

  AffineYieldOpToIntPattern(
      const TypeConverter &typeConverter, MLIRContext *context)
      : mlir::OpConversionPattern<affine::AffineYieldOp>(
            typeConverter, context){};

  LogicalResult matchAndRewrite(affine::AffineYieldOp op,
      typename affine::AffineYieldOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(
        op, adaptor.getOperands());

    return success();
  }
};

void populateAffineYieldOpToIntPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<AffineYieldOpToIntPattern>(typeConverter, ctx);
}

struct AffineLoadOpToIntPattern
    : public OpConversionPattern<affine::AffineLoadOp> {
  using OpConversionPattern<affine::AffineLoadOp>::OpConversionPattern;

  AffineLoadOpToIntPattern(
      const TypeConverter &typeConverter, MLIRContext *context)
      : mlir::OpConversionPattern<affine::AffineLoadOp>(
            typeConverter, context){};

  LogicalResult matchAndRewrite(affine::AffineLoadOp op,
      typename affine::AffineLoadOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto memRefOperand = op.getMemRef();
    auto memRefType = dyn_cast<MemRefType>(memRefOperand.getType());

    if (!memRefType)
      return failure();

    if (!isa<Float32Type>(memRefType.getElementType()))
      return failure();

    Value newMemref = adaptor.getMemref();

    // should we get index instead?
    // this works
    rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
        op, newMemref, op.getMap(), op.getMapOperands());

    return success();
  }
};

void populateAffineLoadOpToIntPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<AffineLoadOpToIntPattern>(typeConverter, ctx);
}

struct AffineStoreOpToIntPattern
    : public OpConversionPattern<affine::AffineStoreOp> {
  using OpConversionPattern<affine::AffineStoreOp>::OpConversionPattern;

  AffineStoreOpToIntPattern(
      const TypeConverter &typeConverter, MLIRContext *context)
      : mlir::OpConversionPattern<affine::AffineStoreOp>(
            typeConverter, context){};

  LogicalResult matchAndRewrite(affine::AffineStoreOp op,
      typename affine::AffineStoreOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto memRefOperand = op.getMemRef();
    auto memRefType = dyn_cast<MemRefType>(memRefOperand.getType());

    if (!memRefType)
      return failure();

    if (!isa<Float32Type>(memRefType.getElementType()))
      return failure();

    // should we use index instead?
    rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
        op, adaptor.getValue(), adaptor.getMemref(), op.getIndices());

    return success();
  }
};

void populateAffineStoreOpToIntPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<AffineStoreOpToIntPattern>(typeConverter, ctx);
}

struct MemRefAllocaOpToIntPattern
    : public OpConversionPattern<memref::AllocaOp> {
  using OpConversionPattern<memref::AllocaOp>::OpConversionPattern;

  MemRefAllocaOpToIntPattern(
      const TypeConverter &typeConverter, MLIRContext *context)
      : mlir::OpConversionPattern<memref::AllocaOp>(typeConverter, context){};

  LogicalResult matchAndRewrite(memref::AllocaOp op,
      typename memref::AllocaOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto memRefType = cast<MemRefType>(op.getType());
    if (!isa<Float32Type>(memRefType.getElementType()))
      return failure();

    auto newMemRefType =
        cast<MemRefType>(getTypeConverter()->convertType(memRefType));

    if (!newMemRefType)
      return failure();

    rewriter.replaceOpWithNewOp<memref::AllocaOp>(
        op, newMemRefType, op.getAlignmentAttr());

    return success();
  }
};

void mlir::populateMemRefAllocaOpToIntPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<MemRefAllocaOpToIntPattern>(typeConverter, ctx);
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
        op.getNameAttrName(), newDenseAttr, op.getOffsetAttr(),
        op.getAlignmentAttr());

    // llvm::errs() << "new op: " << new_op << "\n";
    // llvm::errs() << "uwu" << "\n";

    // log out the original value and the new value
    // for (auto [origValue, newValue] : llvm::zip(
    //          denseAttr.getValues<APFloat>(),
    //          newDenseAttr.getValues<APInt>())) {
    //   llvm::errs() << "original float value: " << origValue.convertToFloat()
    //   << "\n";

    //   llvm::errs() << "original float raw bit: ";
    //   uint64_t orig_raw_bit = origValue.bitcastToAPInt().getZExtValue();
    //   for(int i = 31; i >= 0; i--) {
    //     if (i == 30 || i == 22) {
    //       llvm::errs() << " ";
    //     }
    //     llvm::errs() << ((orig_raw_bit >> i) & 1);
    //   }
    //   llvm::errs() << "\n";

    //   llvm::errs() << "new raw bit: ";
    //   uint64_t raw_bit = newValue.getZExtValue();
    //   for (int i = n_bits - 1; i >= 0; i--) {
    //     llvm::errs() << ((raw_bit >> i) & 1);
    //   }
    //   llvm::errs() << "\n";
    // }

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

    // this only support scalar, return failure if its vector
    if (isa<VectorType>(op->getResult(0).getType()))
      return failure();

    if (!isa<Float32Type>(op.getType()))
      return failure();

    std::string name = getPositFuncStr(n_bits, es_val, opString);

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

void mlir::populateConvertArithAddToPositFuncPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    StringRef opString, uint8_t n_bits, uint8_t es_val) {
  MLIRContext *context = patterns.getContext();
  patterns.add<ConvertArithAddToPositFuncLowering>(
      typeConverter, context, opString, n_bits, es_val);
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
  // populateMemRefAllocaOpToIntPattern(patterns, typeConverter);
  populateMemRefNoOprandToIntPattern<memref::AllocaOp, memref::AllocOp>(
      patterns, typeConverter); // getType() same builder pattern
  populateMemRefLoadOpToIntPattern(patterns, typeConverter);
  populateReinterpretCastOpToIntPattern(patterns, typeConverter);
  populateAffineLoadOpToIntPattern(patterns, typeConverter);
  populateAffineStoreOpToIntPattern(patterns, typeConverter);
  patterns.add<AffineStoreOpToIntPattern>(typeConverter, patterns.getContext());
  populateAffineForOpToIntPattern(patterns, typeConverter);
  populateAffineYieldOpToIntPattern(patterns, typeConverter);

  // populate standard lowering
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);

  ConversionTarget target(getContext());
  target.addIllegalDialect<arith::ArithDialect>();
  target.addDynamicallyLegalOp<arith::ConstantOp>(
      [&](arith::ConstantOp op) { return typeConverter.isLegal(op); });

  target.addDynamicallyLegalOp<KrnlGlobalOp>([&](KrnlGlobalOp op) {
    return typeConverter.isLegal(
        cast<MemRefType>(op->getResult(0).getType()).getElementType());
  });

  target.addDynamicallyLegalOp<memref::AllocaOp, memref::AllocOp>(
      [&](Operation *op) {
        return typeConverter.isLegal(
            cast<MemRefType>(op->getResult(0).getType()));
        // return typeConverter.isLegal(op.getType().getElementType());
      });

  target.addDynamicallyLegalOp<memref::LoadOp, memref::ReinterpretCastOp>(
      [&](Operation *op) { return typeConverter.isLegal(op); });

  target.addDynamicallyLegalOp<affine::AffineLoadOp>(
      [&](Operation *op) { return typeConverter.isLegal(op); });

  target.addDynamicallyLegalOp<affine::AffineStoreOp>(
      [&](affine::AffineStoreOp op) {
        return typeConverter.isLegal(
            cast<MemRefType>(op.getMemref().getType()));
      });

  target.addDynamicallyLegalOp<affine::AffineForOp>(
      [&](affine::AffineForOp op) {
        return typeConverter.isLegal(op->getResultTypes());
      });

  target.addDynamicallyLegalOp<affine::AffineYieldOp>(
      [&](affine::AffineYieldOp op) {
        return typeConverter.isLegal(op->getOperandTypes());
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
