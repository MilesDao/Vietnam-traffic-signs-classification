import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_dir = "rf_dataset/train"
out_dir = "figures"

os.makedirs(out_dir, exist_ok=True)

means = []
stds = []

for cls in os.listdir(img_dir):
    cls_path = os.path.join(img_dir, cls)

    if not os.path.isdir(cls_path):
        continue

    for fn in os.listdir(cls_path):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(cls_path, fn)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        means.append(img.mean(axis=(0, 1)))
        stds.append(img.std(axis=(0, 1)))

means = np.array(means)
stds = np.array(stds)

rgb_mean = means.mean(axis=0)
rgb_std = stds.mean(axis=0)

print("RGB mean:", rgb_mean)
print("RGB std:", rgb_std)

channels = ["R", "G", "B"]
x = np.arange(len(channels))

plt.figure(figsize=(7, 5))
plt.bar(x - 0.2, rgb_mean, width=0.4, label="Mean")
plt.bar(x + 0.2, rgb_std, width=0.4, label="Standard deviation")

plt.xticks(x, channels)
plt.ylim(0, 1)
plt.xlabel("Color channel")
plt.ylabel("Normalized pixel value")
plt.title("Mean and Standard Deviation of RGB Channels after Normalization")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

out_path = os.path.join(out_dir, "rgb_mean_std_normalized.png")
plt.savefig(out_path, dpi=300)
plt.show()

print("Saved figure to:", out_path)