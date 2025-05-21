# snd4onnx \
# --remove_node_names convolution8 activation7 \
# --input_onnx_file_path /home/sylvex/onnx-mlir/experiment/tinyyolov2-7/Model/Model.onnx \
# --output_onnx_file_path /home/sylvex/onnx-mlir/experiment/tinyyolov2-7/Model/del_till_conv7.onnx
# --remove_node_names convolution8 activation7 batchnorm7 convolution7 activation6 batchnorm6 convolution6 pooling5 activation5 batchnorm5 convolution5 pooling4 \

sne4onnx \
-if /home/sylvex/onnx-mlir/experiment/tinyyolov2-7/Model/Model.onnx \
-of /home/sylvex/onnx-mlir/experiment/tinyyolov2-7/Model/del_till_conv7.onnx \
-ion convolution7 \
-oon convolution8