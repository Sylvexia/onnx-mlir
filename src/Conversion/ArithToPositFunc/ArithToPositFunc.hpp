#ifndef MATH_TO_LIB_H_
#define MATH_TO_LIB_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
template <typename T>
class OperationPass;

void populateConvertArithConstantFloatToUIntPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter);
void populateArithToPositFuncConversionPatterns(RewritePatternSet &patterns);

} // namespace mlir

#endif