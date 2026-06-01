import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

feature_dir = "svm_features"
model_path = "svm_models/svm_model.pkl"
results_dir = "svm_results"

os.makedirs(results_dir, exist_ok=True)

print("Loading test features...")
try:
    X_test = np.load(os.path.join(feature_dir, "X_test.npy"))
    y_test = np.load(os.path.join(feature_dir, "y_test.npy"))
except FileNotFoundError:
    print("Error: Feature files not found. Please run prepare_svm_features.py first.")
    exit(1)

print("Loading model...")
if not os.path.exists(model_path):
    print("Error: Model file not found. Please run train_svm.py first.")
    exit(1)
    
svm_model = joblib.load(model_path)

print("Predicting...")
y_pred = svm_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

print("Generating classification report...")
report = classification_report(y_test, y_pred, zero_division=0)
print(report)

report_path = os.path.join(results_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(f"Test Accuracy: {acc:.4f}\n\n")
    f.write(report)

print("Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
np.save(os.path.join(results_dir, "confusion_matrix.npy"), cm)

plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - SVM with Flatten")
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')

cm_img_path = os.path.join(results_dir, "confusion_matrix.png")
plt.savefig(cm_img_path)
print(f"Results saved in '{results_dir}' directory.")
