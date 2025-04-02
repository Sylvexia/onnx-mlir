import matplotlib.pyplot as plt
import os
import json
import math

legends = []
for n_bit in [8, 16]:
    for es in [0, 1, 2, 3]:
        legends.append(f"posit{n_bit}_{es}")

workdir = "/home/sylvex/onnx-mlir/experiment/tinyyolov2-7/output/"
model_name = "tinyyolov2-7"

Average_MAE = []
Average_RMSE = []
for legend in legends:
    # workdir/{legend}/evaluation.json
    evaluation_file = os.path.join(workdir, legend, "evaluation.json")
    with open(evaluation_file, 'r') as f:
        evaluation = json.load(f)
        Average_MAE.append(evaluation['averageMAE'] if not math.isnan(evaluation['averageMAE']) else math.inf)
        Average_RMSE.append(evaluation['averageRMSE'] if not math.isnan(evaluation['averageRMSE']) else math.inf)

# Plot and save Average MAE
plt.figure(figsize=(12, 8))
plt.plot(legends, Average_MAE, marker='o', label='Average MAE')
plt.yscale('log')
plt.legend()
plt.savefig(f'{model_name}_mae.png')

# Plot and save Average RMSE
plt.figure(figsize=(12, 8))
plt.plot(legends, Average_RMSE, marker='o', label='Average RMSE')
plt.yscale('log')
plt.legend()
plt.savefig(f'{model_name}_rmse.png')