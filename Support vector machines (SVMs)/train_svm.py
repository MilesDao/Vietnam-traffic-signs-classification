import os
import time
import joblib
import numpy as np
import scipy.stats as stats

from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

feature_dir = "svm_features"
model_dir = "svm_models"

os.makedirs(model_dir, exist_ok=True)

print("Loading flattened features...")

try:
    X_train = np.load(os.path.join(feature_dir, "X_train.npy"))
    y_train = np.load(os.path.join(feature_dir, "y_train.npy"))
    X_test = np.load(os.path.join(feature_dir, "X_test.npy"))
    y_test = np.load(os.path.join(feature_dir, "y_test.npy"))
except FileNotFoundError:
    print("Error: Feature files not found. Please run prepare_svm_features.py first.")
    exit(1)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

# ==========================================================
# 1. BASELINE MODEL
# ==========================================================
print("\n--- 1. TRAINING BASELINE SVM ---")
baseline_svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight="balanced",
    random_state=42
)

start_base = time.time()
baseline_svm.fit(X_train, y_train)
time_base = time.time() - start_base

y_pred_base = baseline_svm.predict(X_test)
acc_base = accuracy_score(y_test, y_pred_base)
print(f"Baseline Training Time: {time_base:.2f}s")
print(f"Baseline Test Accuracy: {acc_base:.4f}")


# ==========================================================
# 2. RANDOMIZED SEARCH CV
# ==========================================================
print("\n--- 2. RUNNING RANDOMIZED SEARCH CV ---")
# Define parameter space
param_distributions = {
    'C': stats.loguniform(0.1, 100),       # C from 0.1 to 100 on log scale
    'gamma': ['scale', 'auto', 0.01, 0.1], 
    'kernel': ['rbf', 'linear']
}

base_for_search = SVC(class_weight="balanced", random_state=42)

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=base_for_search,
    param_distributions=param_distributions,
    n_iter=10,        # Number of parameter settings that are sampled
    cv=3,             # 3-fold cross-validation
    verbose=2,        # Print progress messages
    n_jobs=-1,        # Use all available cores
    random_state=42
)

start_search = time.time()
random_search.fit(X_train, y_train)
time_search = time.time() - start_search

# Evaluate the best model
best_svm = random_search.best_estimator_
y_pred_tuned = best_svm.predict(X_test)
acc_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"RandomizedSearch Time: {time_search:.2f}s")

# ==========================================================
# 3. COMPARISON REPORT
# ==========================================================
print("\n=======================================================")
print("            HYPERPARAMETER OPTIMIZATION REPORT          ")
print("=======================================================")
print("[BEFORE] Baseline Model")
print(f"   Hyperparameters: C=1.0, kernel='rbf', gamma='scale'")
print(f"   Test Accuracy  : {acc_base:.4f}")
print("-------------------------------------------------------")
print("[AFTER] Tuned Model")
# Format C nicely depending on if it's a float or not
best_c = random_search.best_params_['C']
best_kernel = random_search.best_params_['kernel']
best_gamma = random_search.best_params_['gamma']

print(f"   Hyperparameters: C={best_c:.4f}, kernel='{best_kernel}', gamma='{best_gamma}'")
print(f"   Test Accuracy  : {acc_tuned:.4f}")
print("=======================================================")

# Save the best model
model_path = os.path.join(model_dir, "svm_model.pkl")
joblib.dump(best_svm, model_path)
print(f"\nBest tuned model saved to: {model_path}")
