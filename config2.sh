conda deactivate
deactivate
python -m venv build/env
source ./build/env/bin/activate
MLIR_DIR=/home/sylvex/onnx_llvm/build/lib/cmake/mlir
cmake -G Ninja -B build \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Debug \
    -DPython3_ROOT_DIR=/home/sylvex/onnx-mlir/build/env/bin/python \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_DIR=${MLIR_DIR} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    .

ninja -C build -j 4

pip install joblib
pip install numpy~=1.22.2
pip install -e third_party/onnx
export ONNX_MLIR_HOME=/home/sylvex/onnx-mlir/build/Debug
export CUSTOM_POSIT_LIB_DIR=/home/sylvex/custom_posit/lib
export CUSTOM_POSIT_LIB_NAME=positWrapperC
pip install /home/sylvex/Posit-Numerical-Library
python ./utils/RunONNXModelZooPosit.py -c='-O0' -m='mnist-7' -l='debug' --n-bit="16" --es='2'