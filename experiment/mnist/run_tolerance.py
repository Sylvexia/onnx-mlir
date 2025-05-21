import os
import argparse
import sys
import time
import numpy as np
import json
import libpositWrapperPy as posit

import mnist_dataloader

from onnx import numpy_helper

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--workdir",
        default=os.getcwd(),
        help="Work dir for cloning and downloading, default cwd.",
    )
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

if not os.environ.get("ONNX_MLIR_HOME", None):
    raise RuntimeError(
        "Environment variable ONNX_MLIR_HOME is not set, please set it to the path to "
        "the HOME directory for onnx-mlir. The HOME directory for onnx-mlir refers to "
        "the parent folder containing the bin, lib, etc sub-folders in which ONNX-MLIR "
        "executables and libraries can be found, typically `onnx-mlir/build/Debug`"
    )

getDouble = {
    "8_0": posit.getDouble_8_0,
    "8_1": posit.getDouble_8_1,
    "8_2": posit.getDouble_8_2,
    "8_3": posit.getDouble_8_3,
    "16_0": posit.getDouble_16_0,
    "16_1": posit.getDouble_16_1,
    "16_2": posit.getDouble_16_2,
    "16_3": posit.getDouble_16_3,
    "32_0": posit.getDouble_32_0,
    "32_1": posit.getDouble_32_1,
    "32_2": posit.getDouble_32_2,
    "32_3": posit.getDouble_32_3,
}

getDoubleArray = {
    "8_0": posit.getDoubleArray_8_0,
    "8_1": posit.getDoubleArray_8_1,
    "8_2": posit.getDoubleArray_8_2,
    "8_3": posit.getDoubleArray_8_3,
    "16_0": posit.getDoubleArray_16_0,
    "16_1": posit.getDoubleArray_16_1,
    "16_2": posit.getDoubleArray_16_2,
    "16_3": posit.getDoubleArray_16_3,
    "32_0": posit.getDoubleArray_32_0,
    "32_1": posit.getDoubleArray_32_1,
    "32_2": posit.getDoubleArray_32_2,
    "32_3": posit.getDoubleArray_32_3,
}

getRawBit = {
    "8_0": posit.getRawBit_8_0,
    "8_1": posit.getRawBit_8_1,
    "8_2": posit.getRawBit_8_2,
    "8_3": posit.getRawBit_8_3,
    "16_0": posit.getRawBit_16_0,
    "16_1": posit.getRawBit_16_1,
    "16_2": posit.getRawBit_16_2,
    "16_3": posit.getRawBit_16_3,
    "32_0": posit.getRawBit_32_0,
    "32_1": posit.getRawBit_32_1,
    "32_2": posit.getRawBit_32_2,
    "32_3": posit.getRawBit_32_3,
}

getRawBitArray = {
    "8_0": posit.getRawBitArray_8_0,
    "8_1": posit.getRawBitArray_8_1,
    "8_2": posit.getRawBitArray_8_2,
    "8_3": posit.getRawBitArray_8_3,
    "16_0": posit.getRawBitArray_16_0,
    "16_1": posit.getRawBitArray_16_1,
    "16_2": posit.getRawBitArray_16_2,
    "16_3": posit.getRawBitArray_16_3,
    "32_0": posit.getRawBitArray_32_0,
    "32_1": posit.getRawBitArray_32_1,
    "32_2": posit.getRawBitArray_32_2,
    "32_3": posit.getRawBitArray_32_3,
}

args = get_args()
func_suffix = args.n_bit + "_" + args.es
output_dir = os.path.join(args.workdir, "output_tolerance", f"posit{func_suffix}")
os.makedirs(output_dir, exist_ok=True)

# Include runtime directory in python paths, so PyRuntime can be imported.
RUNTIME_DIR = os.path.join(os.environ["ONNX_MLIR_HOME"], "lib")
sys.path.append(RUNTIME_DIR)

try:
    from PyRuntime import OMExecutionSession
except ImportError:
    raise ImportError(
        "Looks like you did not build the PyRuntime target, build it by running `make PyRuntime`."
        "You may need to set ONNX_MLIR_HOME to `onnx-mlir/build/Debug` since `make PyRuntime` outputs to `build/Debug` by default"
    )

def save_ref(prefix, outputs, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(len(outputs)):
        tensor = numpy_helper.from_array(outputs[i])
        tensor_path = os.path.join(path, f"{prefix}-output_{i}.pb")
        with open(tensor_path, "wb") as f:
            f.write(tensor.SerializeToString())
    print(f"Saved {len(outputs)} outputs to {path}")

def main():
    np.random.seed(42069)
    work_dir = args.workdir
    model = "mnist-8"
    model_dir = os.path.join(work_dir, "model")

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model_name = f"{model}-ground-truth"
    model_name_posit = f"{model}-posit{func_suffix}"
    output_path = os.path.join(model_dir, model_name)
    output_path_posit = os.path.join(model_dir, model_name_posit)
    # Run the Model to Get Ground Truth
    shared_lib_path = output_path + ".so"
    sess = OMExecutionSession(shared_lib_path)

    shared_lib_path_posit = output_path_posit + ".so"
    sess_posit = OMExecutionSession(shared_lib_path_posit)

    num_samples = args.n_sample
    print(f"Running {num_samples} samples")
    first_image, labels = mnist_dataloader.get_random_mnist(1)

    # scale from 2^0 to 2^7, push to images
    images = []
    for i in range(8):
        image = first_image * (2 ** i)
        images.append(image)

    print("Running Ground Truth Model")
    for i, image in enumerate(images):
        inputs = []
        inputs.append(image)

        # get run time
        time_start = time.time()
        outputs = sess.run(inputs)
        time_end = time.time()

        print(f"Ground Truth Model Inference {i}: Time: {time_end - time_start:.2f}s")

        save_ref(f"ground-truth-{i}", outputs, output_dir)
        
        print(f"FP32 Predicted: {np.argmax(outputs)}")

    print("Running Posit Model")
    for i, image in enumerate(images):
        inputs = []
        inputs.append(image)

        posit_inputs = []
        for input in inputs:
            posit_inputs.append(getRawBitArray[func_suffix](input))

        time_start = time.time()
        posit_outs = sess_posit.run(posit_inputs)
        time_end = time.time()

        print(f"Posit Model Inference {i}: Time: {time_end - time_start:.2f}s")

        outputs = []
        for posit_out in posit_outs:
            outputs.append(getDoubleArray[func_suffix](posit_out))

        save_ref(f"posit-{i}", outputs, output_dir)

        print(f"Posit Predicted: {np.argmax(outputs)}")

if __name__ == "__main__":
    main()