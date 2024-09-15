#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "src/Dialect/Posit/PositOps.hpp"

int main() {
  mlir::MLIRContext context;
  context.loadDialect<mlir::PositDialect>();
  auto builder = mlir::OpBuilder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());

  mlir::Type f64Type = builder.getF64Type();
  mlir::FloatAttr f64Attr = builder.getF64FloatAttr(3.14);
  auto lhs = builder.create<mlir::PositConstantOp>(builder.getUnknownLoc(),
    f64Type, f64Attr);

  // auto lhs = builder.create<mlir::PositConstantOp>(builder.getUnknownLoc(),
  //     f64Type, f64Value);
  // auto lhs = builder.create<mlir::PositConstantOp>(builder.getUnknownLoc(),
  // 0.42); auto rhs =
  // builder.create<mlir::PositConstantOp>(builder.getUnknownLoc(), 0.69);

  // auto addOp = builder.create<mlir::PositAddOp>(builder.getUnknownLoc(),
  // 0.38, 0.43);
  module.print(llvm::outs());
  return 0;
}
