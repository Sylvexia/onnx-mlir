import json

avg_ground_truth_time = 0
for n_bit in [8, 16]:
    for es in [0, 1, 2, 3]:
        json_path = f"/home/sylvex/onnx-mlir/experiment/mobilenet-imagenet1000/output/posit{n_bit}_{es}/run_log.json"
        with open(json_path, "r") as f:
            json_data = json.load(f)
        posit_time = 0
        ground_truth_time = 0
        for time in json_data["posit_inference_time"]:
            posit_time += time
        for time in json_data["ground_truth_inference_time"]:
            ground_truth_time += time

        print(f"posit{n_bit}_{es} inference time: {posit_time:.2f} seconds")
        avg_ground_truth_time += ground_truth_time


print(f"Average ground truth inference time: {avg_ground_truth_time / 8:.2f} seconds")