import numpy as np
import onnx
from onnx import numpy_helper

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

def main():
    path1 = "/home/sylvex/onnx-mlir/experiment/mobilenet-imagenet1000/data/ground-truth-0-output_0.pb"
    path2 = "/home/sylvex/onnx-mlir/experiment/mobilenet-imagenet1000/data/posit8_2-0-output_0.pb"

    ref1 = loadref(1, path1)
    ref2 = loadref(1, path2)

    flatten1 = ref1[0].flatten()
    flatten2 = ref2[0].flatten()

    print(f"MAE: {getMAE(flatten1, flatten2)}")
    print(f"RMSE: {getRMSE(flatten1, flatten2)}")

    topk = 5
    print(f"Ground Truth Top {topk} label indices: {getTopKLabelIdx(flatten1, topk)}")
    print(f"Posit8_2 Top {topk} label indices: {getTopKLabelIdx(flatten2, topk)}")

if __name__ == '__main__':
    main()