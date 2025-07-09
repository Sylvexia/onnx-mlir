import numpy as np
import argparse
import json
import onnx
from onnx import numpy_helper

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-bit",
        type=str,
        default="8",
        help="The bit-width for posit data type",
    )
    parser.add_argument(
        "--es",
        type=str,
        default="0",
        help="The exponent size for posit data type",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=1,
        help="Number of samples to run",
    )
    return parser.parse_args()

def getMAE(numpyA, numpyB):
    return np.mean(np.abs(numpyA - numpyB))

def getRMSE(numpyA, numpyB):
    return np.sqrt(np.mean((numpyA - numpyB)**2))

def getTopKLabelIdx(numpyA, k):
    return np.argpartition(numpyA, -k)[-k:]

def loadref(num_inputs, pbfile):
    inputs = []
    for i in range(num_inputs):
        input_ts = onnx.TensorProto()
        with open(pbfile, 'rb') as f:
            input_ts.ParseFromString(f.read())
        input_np = numpy_helper.to_array(input_ts)
        inputs.append(input_np)
    return inputs

def getAccuracies(numpyA, numpyB):
    top1A = getTopKLabelIdx(numpyA, 1)
    top1B = getTopKLabelIdx(numpyB, 1)
    top5A = getTopKLabelIdx(numpyA, 5)
    top5B = getTopKLabelIdx(numpyB, 5)

    top1Acc = len(np.intersect1d(top1A, top1B)) / len(top1A)
    top5Acc = len(np.intersect1d(top5A, top5B)) / len(top5A)

    return top1Acc, top5Acc

def main():
    args = get_args()
    
    MAEs = []
    RMSEs = []
    top1Accuracies = []
    top5Accuracies = []
    matchesLabels = []

    posit_prefix = f"posit{args.n_bit}_{args.es}"
    num_iter = args.n_sample

    dir = f"/home/sylvex/onnx-mlir/experiment/resnet18-imagenet1000/output/{posit_prefix}"
    json_path = f"{dir}/run_log.json"
    real_label = []

    with open(json_path, "r") as f:
        json_data = json.load(f)
        real_label = json_data["labels"]

    for i in range(num_iter):
        print(f"=====Running iteration {i}=====")
        groudPath = f"{dir}/ground-truth-{i}-output_0.pb"
        positPath = f"{dir}/posit-{i}-output_0.pb"

        groundRef = loadref(1, groudPath)
        positRef = loadref(1, positPath)

        flatten1 = groundRef[0].flatten()
        flatten2 = positRef[0].flatten()

        MAE = getMAE(flatten1, flatten2)
        RMSE = getRMSE(flatten1, flatten2)
        MAEs.append(MAE)
        RMSEs.append(RMSE)

        print(f"MAE: {MAE}")
        print(f"RMSE: {RMSE}")

        top1Acc, top5Acc = getAccuracies(flatten1, flatten2)
        top1Accuracies.append(top1Acc)
        top5Accuracies.append(top5Acc)
        matchesLabels.append(real_label[i] == np.argmax(flatten1))

        print(f"Top-1 Accuracy: {top1Acc}")
        print(f"Top-5 Accuracy: {top5Acc}")

    averageMAE = np.mean(MAEs)
    averageRMSE = np.mean(RMSEs)
    averageTop1Accuracy = np.mean(top1Accuracies)
    averageTop5Accuracy = np.mean(top5Accuracies)
    fp32Accuracy = np.mean(matchesLabels)

    print(f"Average MAE: {averageMAE}")
    print(f"Average RMSE: {averageRMSE}")
    print(f"Average Top-1 Accuracy: {averageTop1Accuracy}")
    print(f"Average Top-5 Accuracy: {averageTop5Accuracy}")
    print(f"fp32Accuracy: {fp32Accuracy}")

    json_data = {
        "averageMAE": averageMAE,
        "averageRMSE": averageRMSE,
        "averageTop1Accuracy": averageTop1Accuracy,
        "averageTop5Accuracy": averageTop5Accuracy,
        "fp32Accuracy": fp32Accuracy,
    }
    with open(f"{dir}/evaluation.json", "w") as f:
        json.dump(json_data, f, indent=4)

if __name__ == '__main__':
    main()