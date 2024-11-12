#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

void buildAffineForLoop(MLIRContext *context) {
  OpBuilder builder(context);

  ModuleOp moduleOp = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto memrefType = MemRefType::get({64}, builder.getF32Type());
  auto functionType =
      builder.getFunctionType({memrefType}, {builder.getF32Type()});

  func::FuncOp funcOp = builder.create<func::FuncOp>(
      builder.getUnknownLoc(), "test_affineForLoop", functionType);

  Block *entryBlock = funcOp.addEntryBlock();
  Value memrefArg = entryBlock->getArgument(0);
  builder.setInsertionPointToStart(entryBlock);

  Value initConstant = builder.create<arith::ConstantFloatOp>(
      builder.getUnknownLoc(), APFloat(0.0f), builder.getF32Type());

  auto bodyBuilder = [&](OpBuilder &nestedBuilder, Location loc, Value iv,
                         ValueRange iterArgs) {
    // Load from memref: affine.load %arg0[%arg6]
    auto loadOp = nestedBuilder.create<affine::AffineLoadOp>(
        loc, memrefArg, ValueRange{iv});

    // Add loaded value to accumulator: arith.addf %arg8, %arg7
    Value addResult =
        nestedBuilder.create<arith::AddFOp>(loc, loadOp.getResult(),
            iterArgs[0] // Current accumulator value
        );

    // Yield the result
    nestedBuilder.create<affine::AffineYieldOp>(loc, addResult);
  };

  // Create the affine.for operation
  auto forOp =
      builder.create<affine::AffineForOp>(builder.getUnknownLoc(), // location
          0,            // lower bound
          64,           // upper bound
          1,            // step
          initConstant, // initial value for accumulator
          bodyBuilder   // body builder function
      );

  // print getregion

  // Create return operation
  builder.create<func::ReturnOp>(builder.getUnknownLoc(),
      forOp.getResult(0) // Return the accumulated result
  );

  // Print the module
  moduleOp.print(llvm::outs());
}

// Usage example
int main() {
  MLIRContext context;
  context.loadDialect<affine::AffineDialect, func::FuncDialect,
      memref::MemRefDialect, arith::ArithDialect>();

  buildAffineForLoop(&context);
  return 0;
}