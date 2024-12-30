conda deactivate
deactivate
python -m venv build/env
source ./build/env/bin/activate
pip install joblib
# pip uninstall numpy
pip install numpy~=1.22.2
pip install -e third_party/onnx
export ONNX_MLIR_HOME=/home/sylvex/onnx-mlir/build/Debug
export CUSTOM_POSIT_LIB_DIR=/home/sylvex/custom_posit/lib
export CUSTOM_POSIT_LIB_NAME=positWrapperC
python ./utils/RunONNXModelZooPosit.py -c='-O0' -m='mnist-7' -l='debug' --n-bit="8" --es='2'