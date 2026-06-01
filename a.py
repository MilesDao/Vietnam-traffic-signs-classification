from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Normalize pixel values
scaler = MinMaxScaler()

X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# 2. Apply PCA
pca = PCA(n_components=300, random_state=42)

X_train_pca = pca.fit_transform(X_train_norm)
X_test_pca = pca.transform(X_test_norm)

print("Before PCA:", X_train_norm.shape)
print("After PCA:", X_train_pca.shape)

# 3. Train Random Forest
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_pca, y_train)

# 4. Evaluate
y_pred = rf.predict(X_test_pca)

acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)
print(classification_report(y_test, y_pred))

# 5. Save models
joblib.dump(scaler, "rf_models/scaler.pkl")
joblib.dump(pca, "rf_models/pca.pkl")
joblib.dump(rf, "rf_models/random_forest_pca.pkl")