#ifndef MATH_TO_LIB_H_
#define MATH_TO_LIB_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
template <typename T>
class OperationPass;

void populateConvertArithAddToPositFuncPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, StringRef opString, uint8_t n_bits,
    uint8_t es_val);
void populateConvertArithConstantFloatToIntPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, uint8_t n_bits, uint8_t es_val);
void populateArithToPositFuncConversionPatterns(RewritePatternSet &patterns);

} // namespace mlir

#endif