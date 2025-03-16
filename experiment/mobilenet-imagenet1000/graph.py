import matplotlib.pyplot as plt

legend = ["posit8_0", "posit8_1", "posit8_2", "posit16_0", "posit16_1", "posit16_2", "posit32_0", "posit32_1", "posit32_2"]

Average_MAE = [float("inf"), float("inf"), 2.613018201362924, float("inf"), 0.046334012908511794, 0.017980418850504797, 3.842267976142466e-06, 3.8274386548437175e-06, 3.830007469514384e-06]
Average_RMSE = [float("inf"), float("inf"), 3.4328833767783697, float("inf"), 0.05785140052267175, 0.022920027030189076, 4.893163374052718e-06, 4.879203707754805e-06, 4.880158273937066e-06]
Average_Top1_Accuracy = [0,0,0,0,0.9,1,1,1,1]
Average_Top5_Accuracy = [0,0,0,0,0.96,1,1,1,1]

# Plot and save Average MAE
plt.figure(figsize=(12, 8))
plt.plot(legend, Average_MAE, marker='o', label='Average MAE')
plt.yscale('log')
plt.legend()
plt.savefig('mobilenet-imagenet1000_mae.png')

# Plot and save Average RMSE
plt.figure(figsize=(12, 8))
plt.plot(legend, Average_RMSE, marker='o', label='Average RMSE')
plt.yscale('log')
plt.legend()
plt.savefig('mobilenet-imagenet1000_rmse.png')

# Plot and save Average Top-1 Accuracy
plt.figure(figsize=(12, 8))
plt.plot(legend, Average_Top1_Accuracy, marker='o', label='Average Top-1 Accuracy')
plt.legend()
plt.savefig('mobilenet-imagenet1000_top1_accuracy.png')

# Plot and save Average Top-5 Accuracy
plt.figure(figsize=(12, 8))
plt.plot(legend, Average_Top5_Accuracy, marker='o', label='Average Top-5 Accuracy')
plt.legend()
plt.savefig('mobilenet-imagenet1000_top5_accuracy.png')