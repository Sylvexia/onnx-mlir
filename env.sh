conda deactivate
source build/env/bin/activate
export ONNX_MLIR_HOME=/home/sylvex/onnx-mlir/build/Debug
export CUSTOM_POSIT_LIB_DIR=/home/sylvex/custom_posit/lib
export CUSTOM_POSIT_LIB_NAME=positWrapperC
export LD_LIBRARY_PATH=/home/sylvex/custom_posit/lib:$LD_LIBRARY_PATH