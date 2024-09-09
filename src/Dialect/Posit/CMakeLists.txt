add_onnx_mlir_dialect(Posit posit)
add_onnx_mlir_dialect_doc(posit Posit.td)

add_onnx_mlir_library(OMPositOps
  PositOps.cpp

  DEPENDS
  OMPositIncGen

  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRLLVMCommonConversion
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRShapeDialect
  )
