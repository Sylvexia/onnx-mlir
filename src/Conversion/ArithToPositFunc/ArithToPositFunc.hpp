#ifndef MATH_TO_LIB_H_
#define MATH_TO_LIB_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir {
template <typename T>
class OperationPass;

void populateArithToPositFuncConversionPatterns(RewritePatternSet &patterns);

}

#endif