import matplotlib.pyplot as plt
import os
import json
import math

legends = []
for n_bit in [8, 16]:
    for es in [0, 1, 2, 3]:
        legends.append(f"posit{n_bit}_{es}")

workdir = "/home/sylvex/onnx-mlir/experiment/mobilenet-imagenet1000/output/"
model_name = "mobilenetv2-7"

Average_MAE = []
Average_RMSE = []
Average_Top1_Accuracy = []
Average_Top5_Accuracy = []
for legend in legends:
    # workdir/{legend}/evaluation.json
    evaluation_file = os.path.join(workdir, legend, "evaluation.json")
    with open(evaluation_file, 'r') as f:
        evaluation = json.load(f)
        Average_MAE.append(evaluation['averageMAE'] if not math.isnan(evaluation['averageMAE']) else math.inf)
        Average_RMSE.append(evaluation['averageRMSE'] if not math.isnan(evaluation['averageRMSE']) else math.inf)
        Average_Top1_Accuracy.append(evaluation['averageTop1Accuracy'])
        Average_Top5_Accuracy.append(evaluation['averageTop5Accuracy'])

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

# Plot and save Average Top-1 Accuracy
plt.figure(figsize=(12, 8))
plt.plot(legends, Average_Top1_Accuracy, marker='o', label='Average Top-1 Accuracy')
plt.legend()
plt.savefig(f'{model_name}_top1_accuracy.png')

# Plot and save Average Top-5 Accuracy
plt.figure(figsize=(12, 8))
plt.plot(legends, Average_Top5_Accuracy, marker='o', label='Average Top-5 Accuracy')
plt.legend()
plt.savefig(f'{model_name}_top5_accuracy.png')