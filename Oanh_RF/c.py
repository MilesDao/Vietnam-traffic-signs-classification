import os
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

feature_dir = "rf_features"

X_train = np.load(os.path.join(feature_dir, "X_train.npy"))
y_train = np.load(os.path.join(feature_dir, "y_train.npy"))
X_test = np.load(os.path.join(feature_dir, "X_test.npy"))
y_test = np.load(os.path.join(feature_dir, "y_test.npy"))

depths = [10, 20, 30, None]

for d in depths:
    print("=" * 40)
    print("max_depth:", d)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=d,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    rf.fit(X_train, y_train)

    train_pred = rf.predict(X_train)
    test_pred = rf.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("Train Accuracy:", round(train_acc * 100, 2), "%")
    print("Test Accuracy:", round(test_acc * 100, 2), "%")
    print("Gap:", round((train_acc - test_acc) * 100, 2), "%")