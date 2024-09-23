# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMMathToLibM
  math_to_libm.cpp

  LINK_LIBS PUBLIC
#   MLIRFuncDialect
#   MLIRFuncToLLVM
#   MLIRFuncTransforms
#   MLIRMathToLLVM
#   MLIRMathTransforms
#   MLIRMemRefToLLVM
#   MLIRMemRefTransforms
#   onnx
    MLIRFuncDialect
    MLIRMathDialect
    MLIRLLVMDialect
    MLIRTransforms
    MLIRPass
    MLIRIR
  )