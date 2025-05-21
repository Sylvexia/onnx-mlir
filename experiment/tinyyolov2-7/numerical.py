import numpy as np
import argparse
import json
import math
import onnx
from onnx import numpy_helper

import voc_dataloader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-bit",
        type=str,
        default="16",
        help="The bit-width for posit data type",
    )
    parser.add_argument(
        "--es",
        type=str,
        default="2",
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

anchors = np.array([[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def nms(boxes, iou_threshold=0.4):
    if not boxes:
        return []
    
    # Sort boxes by confidence (descending)
    sorted_boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
    
    keep_boxes = []
    while sorted_boxes:
        # Take the box with highest confidence
        current_box = sorted_boxes.pop(0)
        keep_boxes.append(current_box)
        
        # Compute IoU with remaining boxes
        remaining_boxes = []
        for box in sorted_boxes:
            iou = compute_iou(current_box, box)
            if iou < iou_threshold:
                remaining_boxes.append(box)
        
        # Update sorted_boxes with non-overlapping boxes
        sorted_boxes = remaining_boxes
    
    return keep_boxes

def decode_outputV2(output, anchors, num_classes=20, img_size=416, conf_threshold=0.5, nms_threshold=0.4):
    boxes = []
    stride = img_size / 13
    num_anchors = len(anchors)  # 5

    for i in range(13):
        for j in range(13):
            for k in range(num_anchors):
                # Extract predictions
                base_idx = k * (5 + num_classes)
                tx = output[0, base_idx + 0, i, j]
                ty = output[0, base_idx + 1, i, j]
                tw = output[0, base_idx + 2, i, j]
                th = output[0, base_idx + 3, i, j]
                objectness = output[0, base_idx + 4, i, j]
                class_probs = output[0, base_idx + 5:base_idx + 5 + num_classes, i, j]

                # Transform to bounding box
                bx = (sigmoid(tx) + j) * stride
                by = (sigmoid(ty) + i) * stride
                pw, ph = anchors[k]
                bw = pw * np.exp(tw)
                bh = ph * np.exp(th)

                class_probs = softmax(class_probs)
                class_id = np.argmax(class_probs)

                # Confidence
                conf = sigmoid(objectness) * class_probs[class_id]

                # Store box (x_min, y_min, x_max, y_max, confidence)
                x_min = bx - bw / 2
                y_min = by - bh / 2
                x_max = bx + bw / 2
                y_max = by + bh / 2

                # print(f"Box {len(boxes)}: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}, confidence={conf}")

                if conf > conf_threshold:
                    boxes.append({
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max,
                        'confidence': conf,
                        'class_id': class_id,
                    })

    # sort the boxes by confidence
    boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)

    print(f"Number of boxes before NMS: {len(boxes)}")

    boxes = nms(boxes, nms_threshold)
    print(f"Number of boxes after NMS: {len(boxes)}")

    return boxes

def compute_iou(box1, box2):
    # Extract coordinates
    x1 = max(box1['x_min'], box2['x_min'])
    y1 = max(box1['y_min'], box2['y_min'])
    x2 = min(box1['x_max'], box2['x_max'])
    y2 = min(box1['y_max'], box2['y_max'])
    
    # Compute intersection area
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height
    
    # Compute union area
    box1_area = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
    box2_area = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
    union_area = box1_area + box2_area - inter_area
    
    # Compute IoU
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def calculate_map(predictions, ground_truths, iou_threshold=0.5):
    """
    predictions: List of dicts, each dict: {'image_id', 'class', 'confidence', 'bbox'}
    ground_truths: List of dicts, each dict: {'image_id', 'class', 'bbox'}
    """

    # Step 1: Organize predictions by confidence
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    # Step 2: For each prediction, check if it's a TP or FP
    matched_gt = set()
    tp = []
    fp = []

    for pred in predictions:
        gt_for_image = [gt for gt in ground_truths if gt['image_id'] == pred['image_id'] and gt['class'] == pred['class']]
        
        best_iou = 0
        best_gt = None
        for gt in gt_for_image:
            iou = compute_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt
        
        if best_iou >= iou_threshold and best_gt and best_gt not in matched_gt:
            tp.append(1)
            fp.append(0)
            matched_gt.add(best_gt)
        else:
            tp.append(0)
            fp.append(1)
    
    # Step 3: Compute Precision-Recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = tp_cumsum / (len(ground_truths) + 1e-6)

    # Step 4: Compute AP
    ap = compute_ap(recalls, precisions)
    
    return ap

def compute_ap(recalls, precisions):

    # Add (0,1) start point and (1,0) end point to make curve complete
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # Make the precision envelope (ensure precision is non-increasing)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Find points where recall changes
    indices = np.where(recalls[1:] != recalls[:-1])[0]

    # Sum up (delta recall) * (precision)
    ap = 0.0
    for i in indices:
        ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]

    return ap

def main():
    args = get_args()
    
    MAEs = []
    RMSEs = []

    posit_prefix = f"posit{args.n_bit}_{args.es}"
    num_iter = args.n_sample

    for i in range(num_iter):
        print(f"=====Running iteration {i}=====")
        groudPath = f"/home/sylvex/onnx-mlir/experiment/tinyyolov2-7/output/{posit_prefix}/ground-truth-{i}-output_0.pb"
        positPath = f"/home/sylvex/onnx-mlir/experiment/tinyyolov2-7/output/{posit_prefix}/posit-{i}-output_0.pb"

        groundRef = loadref(1, groudPath)
        positRef = loadref(1, positPath)

        print("printing ground truth")
        gtBoxes = decode_outputV2(groundRef[0], anchors)
        print("printing posit")
        positBoxes = decode_outputV2(positRef[0], anchors)

        map = calculate_map(positBoxes, gtBoxes)
        print(f"mAP: {map}")

        flatten1 = groundRef[0].flatten()
        flatten2 = positRef[0].flatten()

        # print first 10 elements
        for i in range(10):
            print(f"ground truth: {flatten1[i]}")
            print(f"posit: {flatten2[i]}")

        MAE = getMAE(flatten1, flatten2)
        RMSE = getRMSE(flatten1, flatten2)
        MAEs.append(MAE)
        RMSEs.append(RMSE)

        print(f"MAE: {MAE}")
        print(f"RMSE: {RMSE}")

    averageMAE = np.mean(MAEs)
    averageRMSE = np.mean(RMSEs)

    print(f"Average MAE: {averageMAE}")
    print(f"Average RMSE: {averageRMSE}")

    json_data = {
        "averageMAE": averageMAE,
        "averageRMSE": averageRMSE,
    }
    with open(f"/home/sylvex/onnx-mlir/experiment/tinyyolov2-7/output/{posit_prefix}/evaluation.json", "w") as f:
        json.dump(json_data, f, indent=4)

if __name__ == '__main__':
    main()