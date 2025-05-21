import numpy as np
import onnx
from onnx import numpy_helper

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
    tensorPath = f"/home/sylvex/onnx-mlir/experiment/tinyyolov2-7/Model/test_data_set_0/input_0.pb"
    inputs = loadref(1, tensorPath)
    input_data = inputs[0]

    # get average
    avg = np.mean(input_data)
    print(f"Average of input data: {avg}")
    # get max
    max_val = np.max(input_data)
    print(f"Max value of input data: {max_val}")
    # get min
    min_val = np.min(input_data)
    print(f"Min value of input data: {min_val}")
    # print first 10 elements
    print("First 10 elements of input data:")
    for i in range(10):
        print(f"Element {i}: {input_data.flatten()[i]}")


if __name__ == "__main__":
    main()