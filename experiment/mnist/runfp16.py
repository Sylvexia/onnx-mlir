from onnxconverter_common import float16
from onnx import numpy_helper
import argparse
import os
import pathlib
import json

import onnx

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-path",
        required=True,
        help="model path",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=1,
        help="Number of samples to run",
    )
    return parser.parse_args()

def loadref(num_inputs, pbfile):
    inputs = []
    for i in range(num_inputs):
        input_ts = onnx.TensorProto()
        with open(pbfile, 'rb') as f:
            input_ts.ParseFromString(f.read())
        input_np = numpy_helper.to_array(input_ts)
        inputs.append(input_np)
    return inputs

def getMAE(numpyA, numpyB):
    return np.mean(np.abs(numpyA - numpyB))

def getRMSE(numpyA, numpyB):
    return np.sqrt(np.mean((numpyA - numpyB)**2))

def getTopKLabelIdx(numpyA, k):
    return np.argpartition(numpyA, -k)[-k:]

def getAccuracies(numpyA, numpyB):
    top1A = getTopKLabelIdx(numpyA, 1)
    top1B = getTopKLabelIdx(numpyB, 1)
    top5A = getTopKLabelIdx(numpyA, 5)
    top5B = getTopKLabelIdx(numpyB, 5)

    top1Acc = len(np.intersect1d(top1A, top1B)) / len(top1A)
    top5Acc = len(np.intersect1d(top5A, top5B)) / len(top5A)

    return top1Acc, top5Acc

args = get_args()
model_path = args.model_path
model_fp32 = onnx.load(model_path)
model_fp16 = float16.convert_float_to_float16(model_fp32)
# save path is same directory as model_path with "_fp16" suffix
if not model_path.endswith(".onnx"):
    raise ValueError("Model path must end with .onnx extension")

model_path_obj = pathlib.Path(model_path)
save_path = model_path_obj.with_name(model_path_obj.stem + "_fp16.onnx")
print(f"Saving FP16 model to {save_path}")
onnx.save(model_fp16, save_path)

import onnxruntime as ort
import numpy as np
import pathlib
import mnist_dataloader

# Load the FP16 ONNX model
session = ort.InferenceSession("model_fp16.onnx", providers=['CPUExecutionProvider'])

np.random.seed(42069)  # For reproducibility
num_samples = args.n_sample
images, labels = mnist_dataloader.get_random_mnist(num_samples)
matched_count = 0

MAEs = []
RMSEs = []
top1Accuracies = []
top5Accuracies = []
matchesLabels = []
dir = f"/home/sylvex/onnx-mlir/experiment/mnist/output/posit8_0"
json_path = f"{dir}/run_log.json"

with open(json_path, "r") as f:
    json_data = json.load(f)
    real_label = json_data["labels"]

for i, image in enumerate(images):
    fp32Path = f"/home/sylvex/onnx-mlir/experiment/mnist/output/posit8_0/ground-truth-{i}-output_0.pb"
    fp32Ref = loadref(1, fp32Path)

    inputs = {session.get_inputs()[0].name: image.astype(np.float16)}
    fp16out = session.run(None, inputs)
    predicted_label = np.argmax(fp16out[0], axis=1)

    # convert f16 to f32 for comparison
    fp16out = [np.float32(x) for x in fp16out]

    flatten1 = fp32Ref[0].flatten()
    flatten2 = fp16out[0].flatten()

    MAE = getMAE(flatten1, flatten2)
    RMSE = getRMSE(flatten1, flatten2)
    MAEs.append(MAE)
    RMSEs.append(RMSE)

    top1Acc, top5Acc = getAccuracies(flatten1, flatten2)
    top1Accuracies.append(top1Acc)
    top5Accuracies.append(top5Acc)
    matchesLabels.append(real_label[i] == np.argmax(flatten2))

    print(f"Sample {i+1}: Predicted label: {predicted_label[0]}, Actual label: {labels[i]}")
    matched_count += (predicted_label[0] == labels[i])

print(f"Total matched count: {matched_count} out of {num_samples}")
print(f"Accuracy: {matched_count / num_samples * 100:.2f}%")

print(f"Mean Absolute Error (MAE): {np.mean(MAEs)}")
print(f"Root Mean Square Error (RMSE): {np.mean(RMSEs)}")
print(f"Average Top-1 Accuracy: {np.mean(top1Accuracies)}")
print(f"Average Top-5 Accuracy: {np.mean(top5Accuracies)}")

# # Prepare input
# inputs = {session.get_inputs()[0].name: np.random.randn(1, 1, 28, 28).astype(np.float16)}

# # Run inference

# print("Output:", outputs[0])