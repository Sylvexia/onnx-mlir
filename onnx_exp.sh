conda deactivate
source deactivate
python -m venv build/onnx_exp
source ./build/onnx_exp/bin/activate
pip install -U onnx \
&& python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com \
&& pip install -U snd4onnx