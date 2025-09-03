import numpy as np
import argparse
import json
import onnx
from onnx import numpy_helper
import image_net_dataloader as loader

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

def getSSIM(numpyA, numpyB):
    # use numpy nampy to calculate SSIM without skimage
    C1 = 6.5025
    C2 = 58.5225
    mu1 = np.mean(numpyA)
    mu2 = np.mean(numpyB)
    sigma1_sq = np.var(numpyA)
    sigma2_sq = np.var(numpyB)
    sigma12 = np.mean((numpyA - mu1) * (numpyB - mu2))
    ssim = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim

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

def saveImage(array, filename):
    from PIL import Image
    image = array.reshape(3, 224, 224).transpose(1, 2, 0)
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.save(filename)

def main():
    args = get_args()
    np.random.seed(42069)
    
    MAEs = []
    RMSEs = []

    posit_prefix = f"posit{args.n_bit}_{args.es}"
    num_iter = args.n_sample
    dir = f"/home/sylvex/onnx-mlir/experiment/mosaic/output/{posit_prefix}"
    json_path = f"{dir}/run_log.json"

    with open(json_path, "r") as f:
        json_data = json.load(f)

    images, _ = loader.get_random_imagenet(loader.DATAPATH , num_iter)

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

        saveImage(groundRef[0], f"{dir}/ground-truth-{i}.png")
        saveImage(positRef[0], f"{dir}/posit-{i}.png")

    averageMAE = np.mean(MAEs)
    averageRMSE = np.mean(RMSEs)

    print(f"Average MAE: {averageMAE}")
    print(f"Average RMSE: {averageRMSE}")

    json_data = {
        "averageMAE": averageMAE,
        "averageRMSE": averageRMSE,
    }
    with open(f"{dir}/evaluation.json", "w") as f:
        json.dump(json_data, f, indent=4)

if __name__ == '__main__':
    main()