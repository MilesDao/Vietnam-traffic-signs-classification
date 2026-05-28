# Vietnamese Traffic Sign Classification using Random Forest

This project is part of a Vietnamese Traffic Sign Recognition system for the ML2 course.  
This repository focuses on the Random Forest approach for classifying Vietnamese traffic signs.

The model uses cropped traffic sign images as input. Each image is resized, normalized, flattened into a feature vector, and then classified using a Random Forest classifier.

## Project Role

This repository represents the Random Forest branch of the Vietnamese Traffic Sign Recognition project.

The full project compares three different approaches:

- YOLO for object detection
- SVM for traditional machine learning classification
- Random Forest for traditional machine learning classification

In this branch, Random Forest is used to classify cropped traffic sign images after preprocessing.

## Pipeline

The Random Forest branch follows this pipeline:

```txt
Original traffic-sign images
→ Read YOLO-format annotation files
→ Crop traffic signs using bounding boxes
→ Resize cropped signs to 64 × 64
→ Normalize pixel values to [0, 1]
→ Flatten images into feature vectors
→ Train Random Forest classifier
→ Evaluate model performance.  

```

## Important Files

| File | Description |
|---|---|
| `crop_check.py` | Checks whether YOLO bounding boxes are converted correctly and saves sample cropped images for manual inspection. |
| `create_rf_dataset.py` | Reads original images and YOLO labels, crops traffic sign regions, adds 5% padding, and saves cropped images into class folders. |
| `prepare_rf_features.py` | Loads cropped images, resizes them to `64 × 64`, normalizes pixel values to `[0, 1]`, flattens images into feature vectors, and saves them as `.npy` files. |
| `train_random_forest.py` | Loads prepared features, trains a Random Forest classifier, evaluates basic test accuracy, and saves the trained model as a `.pkl` file. |
| `evaluate_random_forest.py` | Loads the trained model, predicts on the test set, generates accuracy, classification report, and confusion matrix. |
| `main.ipynb` | Notebook version used for experimentation, explanation, and step-by-step project workflow. |


## Installation

Install the required Python packages using:

```bash
pip install -r requirements.txt


```
The required packages are:

| Package | Purpose |
|---|---|
| `numpy` | Stores and processes feature arrays |
| `opencv-python` | Reads, resizes, and crops images |
| `scikit-learn` | Trains and evaluates the Random Forest model |
| `matplotlib` | Visualizes the confusion matrix |
| `joblib` | Saves and loads the trained model |




## How to Run

Before running the scripts, make sure the original dataset is placed inside the `archive/` folder.

### Step 1: Check one crop manually

```bash
python crop_check.py
```

This step checks whether YOLO bounding boxes are converted correctly.  
The cropped sample images will be saved in:

```txt
check_crops/
```

### Step 2: Create the cropped Random Forest dataset

```bash
python create_rf_dataset.py
```

This script reads the original images and YOLO label files, crops traffic sign regions, and saves them into:

```txt
rf_dataset/
├── train/
└── test/
```

### Step 3: Prepare feature vectors

```bash
python prepare_rf_features.py
```

This script resizes each cropped image to `64 × 64`, normalizes pixel values to `[0, 1]`, flattens the image into a feature vector, and saves the feature arrays into:

```txt
rf_features/
├── X_train.npy
├── y_train.npy
├── X_test.npy
└── y_test.npy
```

### Step 4: Train the Random Forest model

```bash
python train_random_forest.py
```

This script trains the Random Forest classifier and saves the trained model to:

```txt
rf_models/random_forest_flatten.pkl
```

### Step 5: Evaluate the trained model

```bash
python evaluate_random_forest.py
```

This script evaluates the trained model on the test set and saves the results into:

```txt
rf_results/
├── classification_report.txt
├── confusion_matrix.npy
└── confusion_matrix.png
```

---

## Expected Outputs

After running all scripts successfully, the project will generate:

```txt
check_crops/                 # Sample cropped images for checking
rf_dataset/                  # Cropped train/test dataset
rf_features/                 # NumPy feature arrays
rf_models/                   # Saved Random Forest model
rf_results/                  # Evaluation reports and confusion matrix
```

---

## Results

The trained Random Forest model is evaluated on the test set using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The evaluation report is saved in:

```txt
rf_results/classification_report.txt
```

The confusion matrix image is saved in:

```txt
rf_results/confusion_matrix.png
```

To display the confusion matrix in this README, add:

```markdown
![Confusion Matrix](rf_results/confusion_matrix.png)
```

---

## Demo

A simple demo can be shown using cropped traffic sign images.

Example crop result:

```markdown
![Sample Crop](check_crops/sample_crop.jpg)
```

Note: This Random Forest model requires cropped traffic sign images as input.  
It does not detect traffic signs directly from full road-scene images.

---

## Model Explanation

Random Forest is an ensemble learning algorithm based on multiple decision trees.

Each decision tree learns different rules from the training data.  
The final prediction is made by majority voting from all trees.

In this project, each cropped traffic sign image is converted into a flattened pixel vector.  
The Random Forest model then learns patterns from these pixel values to classify the traffic sign into the correct class.

The model configuration used in this branch is:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
```

---

## Limitations

This Random Forest branch has some limitations:

- Random Forest uses flattened pixel values, so it does not preserve spatial image structure as well as CNN-based models.
- The model requires cropped traffic sign images and cannot detect signs directly from full road-scene images.
- The current version uses fixed hyperparameters without GridSearchCV or RandomizedSearchCV.
- Model performance may decrease if the cropped image is blurry, too small, or incorrectly cropped.
- Class names are currently represented by numeric class IDs such as `class_0`, `class_1`, and so on.

---

## Future Improvements

Possible improvements include:

- Add class name mapping for each traffic sign class.
- Add a prediction script for testing one new cropped traffic sign image.
- Use feature extraction methods such as HOG or color histogram.
- Tune Random Forest hyperparameters using GridSearchCV or RandomizedSearchCV.
- Add a validation set for better model selection.
- Compare Random Forest results with SVM and YOLO results.

---

## Credits

This project was developed as part of the ML2 course project.

This repository contains the Random Forest branch of the group project.

Team approaches:

- YOLO approach: Object detection
- SVM approach: Traditional machine learning classification
- Random Forest approach: Traditional machine learning classification

---

## License

This project is for educational purposes only.