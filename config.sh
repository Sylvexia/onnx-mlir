conda deactivate
deactivate
python -m venv env
source env/bin/activate
MLIR_DIR=/home/sylvex/onnx_llvm/llvm-project/build/lib/cmake/mlir
cmake -G Ninja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_DIR=${MLIR_DIR} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    ..