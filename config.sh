conda deactivate
source deactivate
python -m venv build/env
source ./build/env/bin/activate
pip install joblib
pip install numpy~=1.22.2
pip install -e third_party/onnx
pip install /home/sylvex/Posit-Numerical-Library
pip install pillow
pip install matplotlib~=3.5.0
MLIR_DIR=/home/sylvex/onnx_llvm/llvm-project/build/lib/cmake/mlir
cmake -G Ninja -B build \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_DIR=${MLIR_DIR} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    .

ninja -C build -j 4

export ONNX_MLIR_HOME=/home/sylvex/onnx-mlir/build/Debug
export CUSTOM_POSIT_LIB_DIR=/home/sylvex/custom_posit/lib
export CUSTOM_POSIT_LIB_NAME=positWrapperC
export LD_LIBRARY_PATH=/home/sylvex/custom_posit/lib:$LD_LIBRARY_PATH
python ./utils/RunONNXModelZooPosit.py -c='-O0' -m='mnist-7' -l='debug' --n-bit="32" --es='2'