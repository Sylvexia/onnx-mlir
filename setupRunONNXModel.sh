conda deactivate
source deactivate
python -m venv build/env
source ./build/env/bin/activate
pip install joblib
pip install numpy~=1.22.2
pip install -e third_party/onnx
pip install pillow
export ONNX_MLIR_HOME=/home/sylvex/onnx-mlir/build/Debug
export CUSTOM_POSIT_LIB_DIR=/home/sylvex/custom_posit/lib
export CUSTOM_POSIT_LIB_NAME=positWrapperC
export LD_LIBRARY_PATH=/home/sylvex/custom_posit/lib:$LD_LIBRARY_PATH
pip install /home/sylvex/Posit-Numerical-Library
python ./utils/RunONNXModelZooPosit.py -c='-O0' -m='mnist-7' -l='debug' --n-bit="16" --es='2'