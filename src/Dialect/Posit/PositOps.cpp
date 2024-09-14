#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Value.h"
#include "src/Dialect/Posit/PositOps.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

#define GET_OP_CLASSES
#include "src/Dialect/Posit/PositOps.cpp.inc"
#include "src/Dialect/Posit/PositDialect.cpp.inc"