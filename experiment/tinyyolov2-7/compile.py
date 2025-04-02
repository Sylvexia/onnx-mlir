import numpy as np
import os
import tarfile
import argparse
import time
import copy
import signal
import subprocess
import json

if not os.environ.get("CUSTOM_POSIT_LIB_DIR", None):
    raise RuntimeError(
        "Environment variable CUSTOM_POSIT_LIB_DIR is not set"
    )

if not os.environ.get("CUSTOM_POSIT_LIB_NAME", None):
    raise RuntimeError(
        "Environment variable CUSTOM_POSIT_LIB_NAME is not set"
    )

CURL_CMD = ["curl", "--insecure", "--retry", "50", "--location", "--silent"]

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

def execute_commands(cmds, cwd=None, tmout=None):
    print("cmd={} cwd={}".format(" ".join(cmds), cwd))

    json_data["cmds"].append(" ".join(cmds))

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

if not os.environ.get("ONNX_MLIR_HOME", None):
    raise RuntimeError(
        "Environment variable ONNX_MLIR_HOME is not set, please set it to the path to "
        "the HOME directory for onnx-mlir. The HOME directory for onnx-mlir refers to "
        "the parent folder containing the bin, lib, etc sub-folders in which ONNX-MLIR "
        "executables and libraries can be found, typically `onnx-mlir/build/Debug`"
    )

ONNX_MLIR_EXENAME = "onnx-mlir"
ONNX_MLIR = os.path.join(
    os.environ["ONNX_MLIR_HOME"], "bin", ONNX_MLIR_EXENAME)

args = get_args()
func_suffix = args.n_bit + "_" + args.es
output_dir = os.path.join(args.workdir, "output", f"posit{func_suffix}")
os.makedirs(output_dir, exist_ok=True)
json_log_file = os.path.join(output_dir, "compile_log.json")

json_data = {
    "cmds": [],
    "ground_truth_compile_time": 0,
    "posit_compile_time": 0,
}

def main():
    np.random.seed(42069)
    work_dir = args.workdir
    model_url = "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-7.tar.gz"
    model = "tinyyolov2-7"
    model_tar_gz = os.path.join(work_dir, f"{model}.tar.gz")

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
    model_name = f"{model}-ground-truth"
    model_name_posit = f"{model}-posit{func_suffix}"
    model_dir = os.path.join(work_dir, "model")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    output_path = os.path.join(model_dir, model_name)
    output_path_posit = os.path.join(model_dir, model_name_posit)

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
    json_data["ground_truth_compile_time"] = end - start
    print(f"Normal Compilation time: {end - start:.2f}s")

    start = time.perf_counter()
    ok, msg = execute_commands(command_str_posit)
    if not ok:
        print(msg)
        exit(1)
    end = time.perf_counter()
    json_data["posit_compile_time"] = end - start
    print(f"Posit Compilation time: {end - start:.2f}s")

    with open(json_log_file, "w") as f:
        json.dump(json_data, f, indent=4)

if __name__ == "__main__":
    main()