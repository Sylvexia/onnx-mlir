echo "MNIST" >> modelweight.log
python modelweight.py -m /home/sylvex/onnx-mlir/experiment/mnist/model/model.onnx --name MNIST >> modelweight.txt
echo "mobilenetv2" >> modelweight.log
python modelweight.py -m /home/sylvex/onnx-mlir/experiment/mobilenet-imagenet1000/mobilenetv2-7/mobilenetv2-7.onnx --name mobilenetV2 >> modelweight.txt
echo "resnet18" >> modelweight.log
python modelweight.py -m /home/sylvex/onnx-mlir/experiment/resnet18-imagenet1000/resnet18-v1-7/resnet18-v1-7.onnx --name resnet18 >> modelweight.txt
echo "tinyyolov2-7" >> modelweight.log
python modelweight.py -m /home/sylvex/onnx-mlir/experiment/tinyyolov2-7/Model/Model.onnx --name tinyyolov2 >> modelweight.txt
echo "mosaic" >> modelweight.log
python modelweight.py -m /home/sylvex/onnx-mlir/experiment/mosaic/mosaic/mosaic.onnx --name mosaic >> modelweight.txt