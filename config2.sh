conda deactivate
deactivate
python -m venv env
source env/bin/activate
MLIR_DIR=/home/sylvex/onnx_llvm/build/lib/cmake/mlir
cmake -G Ninja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Debug \
    -DPython3_ROOT_DIR=/home/sylvex/onnx-mlir/build/env/bin/python \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_DIR=${MLIR_DIR} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    ..