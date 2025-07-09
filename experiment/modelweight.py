import onnx
import numpy as np
import argparse

# python modelweight.py -m /home/sylvex/onnx-mlir/experiment/mnist/model/model.onnx
# python modelweight.py -m /home/sylvex/onnx-mlir/experiment/mobilenet-imagenet1000/mobilenetv2-7/mobilenetv2-7.onnx
# python modelweight.py -m /home/sylvex/onnx-mlir/experiment/resnet18-imagenet1000/resnet18-v1-7/resnet18-v1-7.onnx
# python modelweight.py -m /home/sylvex/onnx-mlir/experiment/tinyyolov2-7/Model/Model.onnx

def printHistogram(arr, lower, higher):
    bins = np.arange(lower, higher + 2)
    counts, edges = np.histogram(arr, bins=bins)

    # ASCII Histogram
    max_count = np.max(counts)
    for i, count in enumerate(counts):
        bin_label = f"{int(edges[i]):>3}"
        bar = '#' * (count * 20 // max_count)
        print(f"{bin_label}: {bar}")

def printInfo(arr):
    binExpArr = np.log2(np.abs(arr) + 1e-20)  # Avoid log2(0) by adding a small constant

    print(f"Name: {name}, Shape: {arr.shape}")
    print("Mean:", np.mean(arr), "Std:", np.std(arr), "Max:", np.max(arr), "Min:", np.min(arr))
    print(f"Binary exponent: Mean: {np.mean(binExpArr)}, Std: {np.std(binExpArr)}, Max: {np.max(binExpArr)}, Min: {np.min(binExpArr)}")

    printHistogram(binExpArr, -16, 16)


parser = argparse.ArgumentParser(description="Extract and analyze model weights from an ONNX model.")
parser.add_argument("-m", type=str, required=True, help="Path to the ONNX model file.")
args = parser.parse_args()

model_path = args.m
model = onnx.load(model_path)

print("Model Name:", model.graph.name)
initializers = model.graph.initializer

fullArray  = np.array([], dtype=np.float32)

for tensor in initializers:
    name = tensor.name
    array = onnx.numpy_helper.to_array(tensor)
    fullArray = np.concatenate((fullArray, array.flatten()))

    printInfo(array)

print("Full Array Statistics:")
printInfo(fullArray)
