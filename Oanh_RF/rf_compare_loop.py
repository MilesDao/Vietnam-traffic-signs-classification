import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

feature_dir = "rf_features"
out_dir = "rf_compare_results"

os.makedirs(out_dir, exist_ok=True)

print("Loading features...")

X_train = np.load(os.path.join(feature_dir, "X_train.npy"))
y_train = np.load(os.path.join(feature_dir, "y_train.npy"))
X_test = np.load(os.path.join(feature_dir, "X_test.npy"))
y_test = np.load(os.path.join(feature_dir, "y_test.npy"))

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

n_list = [50, 100, 200, 300]
d_list = [10, 20, 30, None]
f_list = ["sqrt", "log2"]

results = []

for n in n_list:
    for d in d_list:
        for f in f_list:
            print("================================")
            print("Training with:")
            print("n_estimators:", n)
            print("max_depth:", d)
            print("max_features:", f)

            rf = RandomForestClassifier(
                n_estimators=n,
                max_depth=d,
                max_features=f,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )

            start = time.time()

            rf.fit(X_train, y_train)

            end = time.time()

            y_pred = rf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            train_time = end - start

            print("Accuracy:", acc)
            print("Macro F1:", f1)
            print("Training time:", round(train_time, 2), "seconds")

            results.append({
                "n_estimators": n,
                "max_depth": str(d),
                "max_features": f,
                "accuracy": acc,
                "macro_f1": f1,
                "training_time": train_time
            })

df = pd.DataFrame(results)

csv_path = os.path.join(out_dir, "rf_compare_results.csv")
df.to_csv(csv_path, index=False)

print("Saved results to:", csv_path)

best = df.sort_values(by="accuracy", ascending=False).iloc[0]

print("\nBest configuration:")
print(best)

# Chart 1: compare accuracy by n_estimators
plt.figure(figsize=(8, 5))

for f in f_list:
    temp = df[(df["max_depth"] == "None") & (df["max_features"] == f)]
    plt.plot(temp["n_estimators"], temp["accuracy"], marker="o", label=f"max_features={f}")

plt.title("Accuracy Comparison by Number of Trees")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

fig_path = os.path.join(out_dir, "accuracy_by_estimators.png")
plt.savefig(fig_path, dpi=300)
plt.close()

print("Saved:", fig_path)

# Chart 2: compare accuracy by max_depth
plt.figure(figsize=(8, 5))

depth_order = ["10", "20", "30", "None"]

for n in [100, 200]:
    temp = df[(df["n_estimators"] == n) & (df["max_features"] == "sqrt")]
    temp = temp.set_index("max_depth").loc[depth_order].reset_index()
    plt.plot(temp["max_depth"], temp["accuracy"], marker="o", label=f"n_estimators={n}")

plt.title("Accuracy Comparison by Tree Depth")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

fig_path = os.path.join(out_dir, "accuracy_by_depth.png")
plt.savefig(fig_path, dpi=300)
plt.close()

print("Saved:", fig_path)

# Chart 3: compare training time
plt.figure(figsize=(8, 5))

temp = df[(df["max_depth"] == "None") & (df["max_features"] == "sqrt")]
plt.plot(temp["n_estimators"], temp["training_time"], marker="o")

plt.title("Training Time by Number of Trees")
plt.xlabel("n_estimators")
plt.ylabel("Training time (seconds)")
plt.grid(True)
plt.tight_layout()

fig_path = os.path.join(out_dir, "training_time_by_estimators.png")
plt.savefig(fig_path, dpi=300)
plt.close()

print("Saved:", fig_path)