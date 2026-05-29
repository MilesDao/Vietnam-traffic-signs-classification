# Vietnamese Traffic Sign Classification using SVM

This project is part of a Vietnamese Traffic Sign Recognition system.
This repository focuses on the Support Vector Machine (SVM) approach combined with flattened pixel feature extraction for classifying Vietnamese traffic signs.

## Pipeline

The SVM branch follows this pipeline:

```txt
Original traffic-sign images
→ Read YOLO-format annotation files
→ Crop traffic signs using bounding boxes (5% padding)
→ Resize cropped signs to 64 × 64
→ Extract flattened pixel features
→ Train SVM classifier
→ Evaluate model performance.
```

## Important Files

| File | Description |
|---|---|
| `create_svm_dataset.py` | Reads original images and YOLO labels, crops traffic sign regions, adds 5% padding, and saves cropped images into class folders. |
| `prepare_svm_features.py` | Loads cropped images, resizes them to `64 × 64`, normalizes to `[0,1]`, extracts flattened pixel features, and saves them as `.npy` files. |
| `train_svm.py` | Loads flattened features, trains an SVM classifier (baseline & RandomizedSearchCV), evaluates test accuracy, and saves the trained model as a `.pkl` file. |
| `svm_compare_loop.py` | Runs a parameter sweep (Grid Search) across different values of `C`, `gamma`, and `kernel`, saves results to a CSV, and plots visualization charts. |
| `evaluate_svm.py` | Loads the trained model, predicts on the test set, generates accuracy, classification report, and confusion matrix. |

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

Before running the scripts, make sure the original dataset is placed inside the `Vietnam_traffic_sign/` folder at the root directory.

### Step 1: Create the cropped SVM dataset
```bash
cd Minh_SVM
python3 create_svm_dataset.py
```

### Step 2: Prepare flattened features
```bash
python3 prepare_svm_features.py
```

### Step 3: Run Hyperparameter Comparison (Optional)
This script compares different `C` and `kernel` values and outputs visualization charts into `svm_compare_results/`.
```bash
python3 svm_compare_loop.py
```

### Step 4: Train the SVM model
This script uses `RandomizedSearchCV` to find the optimal parameters and saves the best model.
```bash
python3 train_svm.py
```

### Step 5: Evaluate the trained model
This script generates the final classification report and confusion matrix into `svm_results/`.
```bash
python3 evaluate_svm.py
```
