#ifndef MATH_TO_LIB_H_
#define MATH_TO_LIB_H_

#include "mlir/IR/PatternMatch.h"
// #include "mlir/Pass/Pass.h"

namespace mlir {
template <typename T>
class OperationPass;

// #define GEN_PASS_DECL_CONVERTMATHTOLIBM
// #include "mlir/Conversion/Passes.h.inc"

void populateCustomMathToLibmConversionPatterns(RewritePatternSet &patterns);
// std::unique_ptr<Pass> createConvertCustomMathToLibmPass();
}

#endif