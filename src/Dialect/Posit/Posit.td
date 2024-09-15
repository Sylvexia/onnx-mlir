include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Posit_Dialect : Dialect {
  let name = "posit";
  let cppNamespace = "::mlir";
  // let usePropertiesForAttributes = 0;
  // let useDefaultTypePrinterParser = 1;
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect",
  ];
}

// def StringType : Type<CPred<"mlir::isa<posit::StringType>($_self)">, "string type">;

def PositConstantOp : Op<Posit_Dialect, "constant", 
  [Pure]> {
  let summary = "posit const";
  let description = [{
    Posit Constant operation.
  }];
  let arguments = (ins F64:$value);
  let results = (outs F64);

  let builders = [
    OpBuilder<(ins "double":$value)>
  ];
}

def PositAddOp : Op<Posit_Dialect, "add",
  [Pure]> {
  let summary = "posit add";
  let description = [{
    Posit Add operation.
  }];
  let arguments = (ins F64:$lhs, F64:$rhs);
  let results = (outs F64);

  let builders = [
    OpBuilder<(ins "double":$lhs, "double":$rhs)>
  ];
}