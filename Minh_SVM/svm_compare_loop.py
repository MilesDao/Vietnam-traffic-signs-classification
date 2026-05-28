import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

feature_dir = "svm_features"
out_dir = "svm_compare_results"

os.makedirs(out_dir, exist_ok=True)

print("Loading features...")

try:
    X_train = np.load(os.path.join(feature_dir, "X_train.npy"))
    y_train = np.load(os.path.join(feature_dir, "y_train.npy"))
    X_test = np.load(os.path.join(feature_dir, "X_test.npy"))
    y_test = np.load(os.path.join(feature_dir, "y_test.npy"))
except FileNotFoundError:
    print("Error: Feature files not found. Please run prepare_svm_features.py first.")
    exit(1)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# Define parameter lists to compare
c_list = [0.1, 1.0, 10.0, 100.0]
gamma_list = ["scale", "auto"]
kernel_list = ["rbf", "linear"]

results = []

for c in c_list:
    for g in gamma_list:
        for k in kernel_list:
            # linear kernel does not use gamma, skip redundant combinations
            if k == "linear" and g != "scale":
                continue

            print("================================")
            print("Training with:")
            print("C:", c)
            print("gamma:", g)
            print("kernel:", k)

            svm = SVC(
                kernel=k,
                C=c,
                gamma=g,
                random_state=42,
                class_weight="balanced"
            )

            start = time.time()

            svm.fit(X_train, y_train)

            end = time.time()

            y_pred = svm.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            train_time = end - start

            print("Accuracy:", acc)
            print("Macro F1:", f1)
            print("Training time:", round(train_time, 2), "seconds")

            results.append({
                "C": c,
                "gamma": g,
                "kernel": k,
                "accuracy": acc,
                "macro_f1": f1,
                "training_time": train_time
            })

df = pd.DataFrame(results)

csv_path = os.path.join(out_dir, "svm_compare_results.csv")
df.to_csv(csv_path, index=False)

print("\nSaved results to:", csv_path)

best = df.sort_values(by="accuracy", ascending=False).iloc[0]

print("\nBest configuration:")
print(best)

# ---------------------------------------------------------
# Chart 1: Compare accuracy by C (for rbf kernel)
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

for g in gamma_list:
    temp = df[(df["kernel"] == "rbf") & (df["gamma"] == g)]
    plt.plot(temp["C"], temp["accuracy"], marker="o", label=f"gamma={g}")

plt.title("Accuracy Comparison by C parameter (RBF Kernel)")
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.xscale('log')
plt.xticks(c_list, [str(c) for c in c_list])
plt.legend()
plt.grid(True)
plt.tight_layout()

fig_path1 = os.path.join(out_dir, "accuracy_by_c.png")
plt.savefig(fig_path1, dpi=300)
plt.close()

print("Saved:", fig_path1)

# ---------------------------------------------------------
# Chart 2: Compare accuracy by kernel type
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

# We fix gamma="scale" to compare kernels across C
for k in kernel_list:
    temp = df[(df["gamma"] == "scale") & (df["kernel"] == k)]
    plt.plot(temp["C"], temp["accuracy"], marker="o", label=f"kernel={k}")

plt.title("Accuracy Comparison by Kernel Type")
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.xscale('log')
plt.xticks(c_list, [str(c) for c in c_list])
plt.legend()
plt.grid(True)
plt.tight_layout()

fig_path2 = os.path.join(out_dir, "accuracy_by_kernel.png")
plt.savefig(fig_path2, dpi=300)
plt.close()

print("Saved:", fig_path2)

# ---------------------------------------------------------
# Chart 3: Compare training time by C
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

for k in kernel_list:
    temp = df[(df["gamma"] == "scale") & (df["kernel"] == k)]
    plt.plot(temp["C"], temp["training_time"], marker="o", label=f"kernel={k}")

plt.title("Training Time by C parameter")
plt.xlabel("C")
plt.ylabel("Training time (seconds)")
plt.xscale('log')
plt.xticks(c_list, [str(c) for c in c_list])
plt.legend()
plt.grid(True)
plt.tight_layout()

fig_path3 = os.path.join(out_dir, "training_time_by_c.png")
plt.savefig(fig_path3, dpi=300)
plt.close()

print("Saved:", fig_path3)
