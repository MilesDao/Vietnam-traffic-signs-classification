import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

feature_dir = "rf_features"
model_path = "rf_models/random_forest_flatten.pkl"
out_dir = "rf_results"

os.makedirs(out_dir, exist_ok=True)

print("Loading data...")

X_test = np.load(os.path.join(feature_dir, "X_test.npy"))
y_test = np.load(os.path.join(feature_dir, "y_test.npy"))

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("Loading model...")

rf = joblib.load(model_path)

print("Predicting...")

y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)
print("Accuracy percent:", round(acc * 100, 2), "%")

print("\nClassification Report:")
report = classification_report(
    y_test,
    y_pred,
    digits=4,
    zero_division=0
)

print(report)

report_path = os.path.join(out_dir, "classification_report.txt")

with open(report_path, "w") as f:
    f.write("Random Forest Evaluation\n")
    f.write("========================\n\n")
    f.write(f"Accuracy: {acc}\n")
    f.write(f"Accuracy percent: {round(acc * 100, 2)}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("Saved classification report to:", report_path)

cm = confusion_matrix(y_test, y_pred)

cm_path = os.path.join(out_dir, "confusion_matrix.npy")
np.save(cm_path, cm)

print("Saved confusion matrix array to:", cm_path)

plt.figure(figsize=(18, 18))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(
    include_values=False,
    cmap="Blues",
    ax=plt.gca(),
    colorbar=True
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()

fig_path = os.path.join(out_dir, "confusion_matrix.png")
plt.savefig(fig_path, dpi=300)
plt.close()

print("Saved confusion matrix image to:", fig_path)