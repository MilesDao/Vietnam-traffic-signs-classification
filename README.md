# Vietnam Traffic Signs Recognition and Classification 🚦🇻🇳

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Computer Vision](https://img.shields.io/badge/CV-OpenCV-red.svg)](https://opencv.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Vietnam_Traffic_Signs-brightgreen.svg)](#dataset-overview)

This repository hosts the classification branch of a **Vietnamese Traffic Sign Recognition & Classification** system, developed as a machine learning group project. 

The complete project leverages a two-stage approach:
1. **Object Detection**: YOLO is used to locate and output bounding boxes of traffic signs from raw, real-world road scenes.
2. **Traditional Machine Learning Classification**: Using **Support Vector Machines (SVMs)** and **Random Forest Classifiers** on cropped sign images to classify them into their respective traffic sign types.

---

## 📂 Repository Structure

The project has been structured into dedicated modules representing each classification approach:

```txt
Vietnam-traffic-signs-classification/
│
├── Support vector machines (SVMs)/     # SVM Classification pipeline
│   ├── create_svm_dataset.py            # Crop traffic signs from raw images with 5% padding
│   ├── prepare_svm_features.py          # Resize (64x64), normalize, and extract flat features
│   ├── train_svm.py                    # Train baseline SVM & perform RandomizedSearchCV
│   ├── svm_compare_loop.py              # Exhausive hyperparameter sweep & visualization
│   ├── evaluate_svm.py                 # Load trained models & run evaluation metrics
│   ├── requirements.txt                 # Project dependencies
│   ├── REPORT.md                        # Detailed SVM performance analysis & report
│   └── README.md                        # SVM-specific setup & run guide
│
├── RandomForestClassifier/             # Random Forest Classification pipeline
│   └── REPORT.md                        # Documentation & reports for Random Forest
│
├── Vietnam_traffic_sign/               # [Gitignored] Raw dataset folder
│   ├── images/                          # Raw traffic scene images (.jpg)
│   ├── labels/                          # YOLO bounding boxes annotations (.txt)
│   ├── classes.txt                      # 52 class index labels
│   ├── classes_en.txt                   # Class names in English
│   └── classes_vie.txt                  # Class names in Vietnamese
│
└── .gitignore                          # Excludes raw data and local cache directories
```

---

## 📊 Dataset Overview

The system classifies traffic signs into **52 distinct classes** commonly found on Vietnamese roads, including speed limits, warning signs, and regulatory markers.

- **Data format**: Raw images (`Vietnam_traffic_sign/images/`) and their corresponding YOLO annotation text files (`Vietnam_traffic_sign/labels/`) containing coordinates in the standard format:
  ```txt
  <class_id> <x_center> <y_center> <width> <height>
  ```
- **Languages**: Mappings are provided in both **English** (`classes_en.txt`) and **Vietnamese** (`classes_vie.txt`).

---

## ⚙️ Data Preprocessing & Feature Extraction

Traditional classifiers (SVM, Random Forest) require uniform feature inputs. The pipeline standardizes raw images using the following sequence:

```mermaid
graph TD
    A[Raw Road Scenes + YOLO Annotations] --> B[Crop Bounding Box + 5% Padding]
    B --> C[Resize to 64 × 64 Pixels]
    C --> D[Normalize Intensities to [0,1]]
    D --> E[Flatten into 1D Vector - 12,288 features]
    E --> F[Train Classifiers]
```

1. **Context-Aware Cropping**: Bounding boxes are cropped from raw images, adding a **5% padding** to capture surrounding boundary lines, helping the classifier recognize the sign edges.
2. **Resolution Standardization**: All crops are resized to a uniform `64 × 64` size.
3. **Normalization**: Pixel values are normalized from `[0, 255]` to `[0.0, 1.0]`.
4. **Vectorization**: Resized RGB images are flattened into a 1D feature vector of shape `(12288,)` ($64 \times 64 \times 3$).

---

## 🧠 Model Architectures & Results

### 1. Support Vector Machines (SVMs)
Using standard flattened pixels, an exhaustive parameter sweep was performed across regularizer weight $C$, kernel types, and class weighting to handle high dimensionality.

*   **Optimal Configuration**:
    - **Kernel**: Radial Basis Function (`rbf`)
    - **Regularization ($C$)**: `10.0`
    - **Gamma**: `scale`
    - **Class Weights**: `balanced` (Crucial for handling heavily skewed class distributions)
*   **Performance Achievements**:
    - **Test Accuracy**: **`96.72%`** (An improvement over the `91.00%` baseline SVM)
    - **Macro F1-Score**: **`95.39%`** (Ensures reliable performance across both high-frequency and rare traffic sign classes)

Detailed SVM reports and learning curves are saved under `Support vector machines (SVMs)/REPORT.md`.

### 2. Random Forest Classifier
An ensemble configuration designed for resilient decision boundary splits across raw features.

*   **Model Configuration**:
    - `n_estimators`: `300` (Number of decision trees)
    - `max_depth`: `30`
    - `max_features`: `sqrt`
    - `class_weight`: `balanced`

---

## 🚀 Getting Started

### 📋 Prerequisites & Installation

Clone this repository to your local machine, and ensure you have Python 3.8+ installed.

Install the required dependencies using pip:
```bash
pip install -r "Support vector machines (SVMs)/requirements.txt"
```

### 📥 Dataset Setup

Ensure that the raw traffic sign dataset is placed under the root directory inside `Vietnam_traffic_sign/` as shown in the directory structure:
```txt
Vietnam-traffic-signs-classification/
└── Vietnam_traffic_sign/
    ├── images/
    ├── labels/
    ├── classes.txt
    ├── classes_en.txt
    └── classes_vie.txt
```
*(Note: This folder is automatically ignored by Git to avoid uploading heavy data files).*

---

## 💻 How to Run the SVM Branch

To run the full SVM preprocessing, hyperparameter search, and evaluation pipeline:

### Step 1: Crop and split the dataset
Reads raw images and YOLO bounding boxes, crops the sign regions with a 5% margin, splits them into `train/` and `test/` sets, and organizes them inside class folders.
```bash
cd "Support vector machines (SVMs)"
python create_svm_dataset.py
```

### Step 2: Extract feature vectors
Normalizes, resizes, and flattens the cropped images into `.npy` feature files:
```bash
python prepare_svm_features.py
```

### Step 3: Run hyperparameter sweep (Optional)
Executes a grid sweep comparing `C` parameters, kernel types, and saves comparison charts and `svm_compare_results.csv` under `svm_compare_results/`.
```bash
python svm_compare_loop.py
```

### Step 4: Train the optimal model
Finds optimal parameters using randomized search, trains the final classifier on the full training set, and stores the serialized model as a `.pkl` file.
```bash
python train_svm.py
```

### Step 5: Evaluate the model
Loads the trained model, runs predictions on the test set, and outputs a classification report and confusion matrix visualization under `svm_results/`.
```bash
python evaluate_svm.py
```

---

## 🎯 Future Enhancements

*   **Advanced Feature Descriptors**: Introduce **Histogram of Oriented Gradients (HOG)** or Color Histograms to capture edge directions and color layouts rather than raw pixel intensities, increasing robustness to varying illumination.
*   **Convolutional Neural Networks (CNNs)**: Shift from classical machine learning models to deep learning architectures (e.g., ResNet, MobileNet) to automatically learn spatial hierarchies and local pixel correlations.
*   **Pipeline Integration**: Connect the YOLO detection module directly with the trained classifier to achieve end-to-end inference on raw video frames.

---

## 👥 Credits

This repository is part of a collaborative academic ML2 project.
*   **SVM Implementation & Evaluation**: [Minh-NK](https://github.com/MilesDao)
*   **Random Forest Implementation**: ML2 Course Team Members
