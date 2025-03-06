
# curl --insecure --retry 50 
# --location --silent https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.tar.gz 
# --time-cond /home/sylvex/onnx-mlir/mobilenetv2-7.tar.gz 
# --output /home/sylvex/onnx-mlir/mobilenetv2-7.tar.gz

import os
import argparse
import logging
import sys
import subprocess
import signal
import tarfile
import time
import numpy as np
import json
import copy

import libpositWrapperPy as posit
import image_net_dataloader

from onnx import numpy_helper

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--compile-args", help="Options passing to onnx-mlir to compile a model."
    )
    parser.add_argument(
        "-l",
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="log level, default info",
    )
    parser.add_argument(
        "--log-to-file",
        action="store",
        nargs="?",
        const="compilation.log",
        default=None,
        help="Output compilation messages to file, default compilation.log",
    )
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
        default="2",
        help="The exponent size for posit data type",
    )
    return parser.parse_args()

if not os.environ.get("ONNX_MLIR_HOME", None):
    raise RuntimeError(
        "Environment variable ONNX_MLIR_HOME is not set, please set it to the path to "
        "the HOME directory for onnx-mlir. The HOME directory for onnx-mlir refers to "
        "the parent folder containing the bin, lib, etc sub-folders in which ONNX-MLIR "
        "executables and libraries can be found, typically `onnx-mlir/build/Debug`"
    )

if not os.environ.get("CUSTOM_POSIT_LIB_DIR", None):
    raise RuntimeError(
        "Environment variable CUSTOM_POSIT_LIB_DIR is not set"
    )

if not os.environ.get("CUSTOM_POSIT_LIB_NAME", None):
    raise RuntimeError(
        "Environment variable CUSTOM_POSIT_LIB_NAME is not set"
    )

LOG_LEVEL = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# log to stderr so that stdout can be used for check results
def get_logger():
    logging.basicConfig(
        stream=sys.stderr,
        level=LOG_LEVEL[args.log_level],
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    return logging.getLogger("RunONNXModelZoo.py")

args = get_args()
logger = get_logger()

def execute_commands(cmds, cwd=None, tmout=None):
    logger.debug("cmd={} cwd={}".format(" ".join(cmds), cwd))
    out = subprocess.Popen(
        cmds, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        stdout, stderr = out.communicate(timeout=tmout)
    except subprocess.TimeoutExpired:
        # Kill the child process and finish communication
        out.kill()
        stdout, stderr = out.communicate()
        return (
            False,
            (
                stderr.decode("utf-8")
                + stdout.decode("utf-8")
                + "Timeout after {} seconds".format(tmout)
            ),
        )
    msg = stderr.decode("utf-8") + stdout.decode("utf-8")
    if out.returncode == -signal.SIGSEGV:
        return (False, msg + "Segfault")
    if out.returncode != 0:
        return (False, msg + "Return code {}".format(out.returncode))
    return (True, stdout.decode("utf-8"))

CURL_CMD = ["curl", "--insecure", "--retry", "50", "--location", "--silent"]
ONNX_MLIR_EXENAME = "onnx-mlir"
ONNX_MLIR = os.path.join(
    os.environ["ONNX_MLIR_HOME"], "bin", ONNX_MLIR_EXENAME)
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

func_suffix = args.n_bit + "_" + args.es

def get_names_in_signature(signature):
    names = []
    # Load the input signature.
    signature_dict = json.loads(signature)
    for sig in signature_dict:
        names.append(sig["name"])
    return names

def save_ref(prefix, outputs, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(len(outputs)):
        tensor = numpy_helper.from_array(outputs[i])
        tensor_path = os.path.join(path, f"{prefix}-output_{i}.pb")
        with open(tensor_path, "wb") as f:
            f.write(tensor.SerializeToString())

def main():
    np.random.seed(69420)
    model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.tar.gz"
    work_dir = args.workdir
    model_tar_gz = os.path.join(work_dir, "mobilenetv2-7.tar.gz")
    
    ok, _ = execute_commands(
        CURL_CMD + [model_url, "--time-cond",
                    model_tar_gz, "--output", model_tar_gz],
        cwd=work_dir,
    )

    with tarfile.open(model_tar_gz, "r:gz") as tgz:
        tgz.extractall(work_dir)

    _, onnx_files = execute_commands(
            ["find", work_dir, "-type", "f", "-name", "[^.]*.onnx"]
    )

    onnx_file = onnx_files.split("\n")[0]
    model_name = "mobilenetv2-7-ground-truth"
    model_name_posit = f"mobilenetv2-7-posit{func_suffix}"
    model_dir = os.path.join(work_dir, "model")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    output_path = os.path.join(model_dir, model_name)
    output_path_posit = os.path.join(work_dir, "model", model_name_posit)
    
    # Compile the model
    command_str = [ONNX_MLIR]
    command_str += [onnx_file]

    command_str_posit = copy.deepcopy(command_str)
    command_str_posit += ["--enable-posit", f"--n-bits={args.n_bit}", f"--es-val={args.es}"]
    command_str_posit += [f"-L{os.environ['CUSTOM_POSIT_LIB_DIR']}",
                    f"-l{os.environ['CUSTOM_POSIT_LIB_NAME']}"]
    
    command_str += ["-o", output_path]
    command_str_posit += ["-o", output_path_posit]

    start = time.perf_counter()
    ok, msg = execute_commands(command_str)
    if not ok:
        print(msg)
        exit(1)
    end = time.perf_counter()
    logger.info(f"Normal Compilation time: {end - start:.2f}s")

    start = time.perf_counter()
    ok, msg = execute_commands(command_str_posit)
    if not ok:
        print(msg)
        exit(1)
    end = time.perf_counter()
    logger.info(f"Posit Compilation time: {end - start:.2f}s")

    # Run the Model to Get Ground Truth
    shared_lib_path = output_path + ".so"
    sess = OMExecutionSession(shared_lib_path)

    shared_lib_path_posit = output_path_posit + ".so"
    sess_posit = OMExecutionSession(shared_lib_path_posit)

    num_samples = 1
    images, labels = image_net_dataloader.get_random_imagenet(
        image_net_dataloader.DATAPATH, num_samples
    )

    for i, image in enumerate(images):
        inputs = []
        inputs.append(image)
        outputs = sess.run(inputs)

        save_ref(f"ground-truth-{i}", outputs, os.path.join(work_dir, "data"))
        
        print(f"Ground Truth: {labels[i]}")
        print(f"Predicted: {np.argmax(outputs)}")

    print("Running Posit Model")
    for i, image in enumerate(images):
        inputs = []
        inputs.append(image)

        posit_inputs = []
        for input in inputs:
            posit_inputs.append(getRawBitArray[func_suffix](input))

        posit_outs = sess_posit.run(inputs)

        outputs = []
        for posit_out in posit_outs:
            outputs.append(getDoubleArray[func_suffix](posit_out))

        save_ref(f"posit{func_suffix}-{i}", outputs, os.path.join(work_dir, "data"))
        print(outputs)

        print(f"Ground Truth: {labels[i]}")
        print(f"Predicted: {np.argmax(outputs)}")

if __name__ == "__main__":
    main()