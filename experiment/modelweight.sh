echo "MNIST" >> modelweight.log
python modelweight.py -m /home/sylvex/onnx-mlir/experiment/mnist/model/model.onnx >> modelweight.log
echo "mobilenetv2" >> modelweight.log
python modelweight.py -m /home/sylvex/onnx-mlir/experiment/mobilenet-imagenet1000/mobilenetv2-7/mobilenetv2-7.onnx >> modelweight.log
echo "resnet18" >> modelweight.log
python modelweight.py -m /home/sylvex/onnx-mlir/experiment/resnet18-imagenet1000/resnet18-v1-7/resnet18-v1-7.onnx >> modelweight.log
echo "tinyyolov2-7" >> modelweight.log
python modelweight.py -m /home/sylvex/onnx-mlir/experiment/tinyyolov2-7/Model/Model.onnx >> modelweight.log