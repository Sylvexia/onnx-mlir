add_onnx_mlir_library(OMArithToPositFunc
  ArithToPositFunc.cpp

  LINK_LIBS PUBLIC
#   MLIRFuncDialect
#   MLIRFuncToLLVM
#   MLIRFuncTransforms
#   MLIRMathToLLVM
#   MLIRMathTransforms
#   MLIRMemRefToLLVM
#   MLIRMemRefTransforms
#   onnx
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRTransforms
    MLIRPass
    MLIRIR
  )