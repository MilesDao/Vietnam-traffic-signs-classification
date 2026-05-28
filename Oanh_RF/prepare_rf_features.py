import os
import cv2
import numpy as np

img_size = 64

train_dir = "rf_dataset/train"
test_dir = "rf_dataset/test"

out_dir = "rf_features"
os.makedirs(out_dir, exist_ok=True)


def load_data(data_dir):
    X = []
    y = []

    class_folders = sorted(os.listdir(data_dir))

    for class_name in class_folders:
        class_path = os.path.join(data_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        if not class_name.startswith("class_"):
            continue

        label = int(class_name.replace("class_", ""))

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            img = cv2.imread(img_path)

            if img is None:
                print("Cannot read:", img_path)
                continue

            img = cv2.resize(img, (img_size, img_size))

            img = img / 255.0

            feature = img.flatten()

            X.append(feature)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y


X_train, y_train = load_data(train_dir)
X_test, y_test = load_data(test_dir)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

np.save(os.path.join(out_dir, "X_train.npy"), X_train)
np.save(os.path.join(out_dir, "y_train.npy"), y_train)
np.save(os.path.join(out_dir, "X_test.npy"), X_test)
np.save(os.path.join(out_dir, "y_test.npy"), y_test)

print("Saved features to:", out_dir)