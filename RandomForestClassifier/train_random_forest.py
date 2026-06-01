import os
import time
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

feature_dir = "rf_features"
model_dir = "rf_models"

os.makedirs(model_dir, exist_ok=True)

print("Loading features...")

X_train = np.load(os.path.join(feature_dir, "X_train.npy"))
y_train = np.load(os.path.join(feature_dir, "y_train.npy"))
X_test = np.load(os.path.join(feature_dir, "X_test.npy"))
y_test = np.load(os.path.join(feature_dir, "y_test.npy"))

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

start = time.time()

rf.fit(X_train, y_train)

end = time.time()

print("Training done.")
print("Training time:", round(end - start, 2), "seconds")

print("Predicting...")

y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("Test Accuracy:", acc)

model_path = os.path.join(model_dir, "random_forest_flatten.pkl")
joblib.dump(rf, model_path)

print("Model saved to:", model_path)