#ifndef POSIT_OPS
#define POSIT_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Posit_Dialect : Dialect {
  let name = "posit";
  let cppNamespace = "::mlir";
  let usePropertiesForAttributes = 0;
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect",
    "linalg::LinalgDialect",    // remove
    "math::MathDialect",        // remove
  ];
}

def PositAddOp : Op<Posit_Dialect, "add",
  [Pure]> {
  let summary = "posit add";
  let description = [{
    Posit Add operation.
  }];
  let arguments = (ins P8E0:$lhs, P8E0:$rhs);
  let results = (outs P8E0);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

#endif // POSIT_OPS