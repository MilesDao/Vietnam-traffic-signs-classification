# Report: Traffic Sign Classification using Random Forest

## 1. Introduction

This report presents the Random Forest branch of the traffic sign classification project.
The goal of this part is to classify cropped traffic sign images into their correct classes using a traditional machine learning model.

Random Forest is used because it is simple, explainable, and suitable as a classical machine learning approach for image classification after feature extraction. Since Random Forest cannot directly process full road-scene images like an object detection model, the input images are first cropped using the provided annotation files. Then, the cropped traffic sign images are resized, normalized, flattened, and used as input features for the classifier.



## 2. Objective

The main objective of this part is to build and evaluate a Random Forest model for traffic sign classification.

The specific objectives are:

* Convert annotated traffic sign regions into cropped images.
* Preprocess cropped images into fixed-size numerical feature vectors.
* Train a Random Forest classifier.
* Evaluate the model using accuracy, classification report, and confusion matrix.
* Analyze the limitations and possible improvements of the Random Forest approach.

---

## 3. Dataset and Input Format

The dataset contains original traffic sign images and corresponding annotation files.
Each annotation file stores the class ID and bounding box information of traffic signs in the image.

The annotation format is:

```txt
class_id x_center y_center width height
```

All bounding box values are normalized between `0` and `1`.
Before cropping, these normalized values are converted into pixel coordinates based on the width and height of the original image.

The original dataset structure is:

```txt
archive/
├── images/
├── labels/
└── split_dataset/
    ├── train_files.txt
    └── test_files.txt
```

The training and testing split used in this branch is:

```txt
Training images: 2552
Testing images: 639
```

> Note: The number of cropped images may be different from the number of original images because one image can contain multiple traffic signs, while some images may be skipped if their label files are missing or empty.

---

## 4. Preprocessing Pipeline

The preprocessing stage is important because Random Forest requires numerical feature vectors instead of raw image files.

The preprocessing pipeline follows these steps:

```txt
Original images
→ Read annotation files
→ Convert bounding boxes to pixel coordinates
→ Crop traffic sign regions
→ Resize cropped images to 64 × 64
→ Normalize pixel values to [0, 1]
→ Flatten images into feature vectors
→ Train Random Forest classifier
```

### Figure 1. Random Forest Pipeline

![Random Forest Pipeline](figures/fig1_pipeline.png)

The cropping process uses the bounding box from the annotation file.
A small padding of `5%` is added around each bounding box to avoid cutting important parts of the traffic sign.

After cropping, each image is resized to:

```txt
64 × 64 pixels
```

Then, pixel values are normalized by dividing by `255.0`, so the value range becomes:

```txt
[0, 1]
```

Finally, each RGB image is flattened into a one-dimensional feature vector:

```txt
64 × 64 × 3 = 12288 features
```

This vector is used as the input for the Random Forest model.

---

## 5. Cropped Dataset

After preprocessing, the cropped traffic sign images are saved into class folders.

The output cropped dataset structure is:

```txt
rf_dataset/
├── train/
│   ├── class_0/
│   ├── class_1/
│   └── ...
└── test/
    ├── class_0/
    ├── class_1/
    └── ...
```

Each folder represents one traffic sign class.

### Figure 2. Example Original Image and Bounding Box

![Original Image with Bounding Box](figures/fig2_original_bbox.png)

### Figure 3. Example Cropped Traffic Signs

![Cropped Traffic Signs](figures/fig3_cropped_examples.png)

The cropped images are used as the direct input samples for the Random Forest classifier.

---

## 6. Feature Extraction

Random Forest cannot learn directly from image matrices in the same way as deep learning models.
Therefore, each cropped image must be converted into a numerical feature vector.

In this branch, raw pixel values are used as features.

The feature extraction process is:

```txt
Cropped RGB image
→ Resize to 64 × 64
→ Normalize pixel values
→ Flatten to 12288-dimensional vector
```

The generated feature files are saved as:

```txt
rf_features/
├── X_train.npy
├── y_train.npy
├── X_test.npy
└── y_test.npy
```

Where:

* `X_train.npy` contains training feature vectors.
* `y_train.npy` contains training labels.
* `X_test.npy` contains testing feature vectors.
* `y_test.npy` contains testing labels.

### Figure 4. Image Flattening Process

![Image Flattening Process](figures/fig4_flattening.png)

This feature representation is simple and easy to implement.
However, it does not preserve the spatial structure of images very well, which is one limitation of using Random Forest for image classification.

---

## 7. Random Forest Model

Random Forest is an ensemble learning algorithm based on multiple decision trees.
Each decision tree learns a set of decision rules from the training data.
During prediction, all trees vote for a class, and the final class is selected by majority voting.

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

The meaning of each parameter is shown below:

| Parameter                 | Meaning                                                  |
| ------------------------- | -------------------------------------------------------- |
| `n_estimators=100`        | The model uses 100 decision trees.                       |
| `max_depth=None`          | Trees can grow until fully expanded.                     |
| `max_features="sqrt"`     | Each split considers the square root number of features. |
| `random_state=42`         | Makes the experiment reproducible.                       |
| `n_jobs=-1`               | Uses all available CPU cores for faster training.        |
| `class_weight="balanced"` | Helps reduce the effect of class imbalance.              |

The trained model is saved as:

```txt
rf_models/random_forest_flatten.pkl
```

---

## 8. Hyperparameter Comparison

To understand how different settings affect model performance, several Random Forest hyperparameters can be compared.

The main hyperparameters are:

| Hyperparameter | Meaning                                     | Expected Effect                                               |
| -------------- | ------------------------------------------- | ------------------------------------------------------------- |
| `n_estimators` | Number of trees                             | More trees can improve stability but increase training time.  |
| `max_depth`    | Maximum depth of each tree                  | Deeper trees can learn more complex patterns but may overfit. |
| `max_features` | Number of features considered at each split | Controls randomness between trees.                            |
| `class_weight` | Class balancing strategy                    | Helps when some classes have fewer samples.                   |

Example experiment table:

| Experiment | `n_estimators` | `max_depth` | `max_features` | `class_weight` |      Accuracy |
| ---------- | -------------: | ----------: | -------------- | -------------- | ------------: |
| RF-1       |             50 |          10 | sqrt           | balanced       | To be updated |
| RF-2       |            100 |          20 | sqrt           | balanced       | To be updated |
| RF-3       |            100 |        None | sqrt           | balanced       | To be updated |
| RF-4       |            200 |        None | sqrt           | balanced       | To be updated |
| RF-5       |            100 |        None | log2           | balanced       | To be updated |

### Figure 5. Hyperparameter Accuracy Comparison

![Hyperparameter Accuracy Comparison](figures/fig5_hyperparameter_comparison.png)

From these experiments, the final model configuration can be selected based on the balance between accuracy and training time.

In the current version, `n_estimators=100` and `max_features="sqrt"` are used because they provide a reasonable balance between performance and computational cost.

---

## 9. Evaluation Metrics

The trained Random Forest model is evaluated on the test set.

The evaluation metrics include:

| Metric           | Meaning                                                     |
| ---------------- | ----------------------------------------------------------- |
| Accuracy         | Overall percentage of correct predictions.                  |
| Precision        | How many predicted samples of a class are actually correct. |
| Recall           | How many true samples of a class are correctly predicted.   |
| F1-score         | Balance between precision and recall.                       |
| Confusion Matrix | Shows correct and incorrect predictions between classes.    |

The evaluation outputs are saved in:

```txt
rf_results/
├── classification_report.txt
├── confusion_matrix.npy
└── confusion_matrix.png
```

---

## 10. Results

The model is tested using the prepared test feature vectors.

Current result:

```txt
Test Accuracy: To be updated
```

The classification report contains precision, recall, and F1-score for each class.

Example result format:

| Class   |     Precision |        Recall |      F1-score |       Support |
| ------- | ------------: | ------------: | ------------: | ------------: |
| class_0 | To be updated | To be updated | To be updated | To be updated |
| class_1 | To be updated | To be updated | To be updated | To be updated |
| class_2 | To be updated | To be updated | To be updated | To be updated |

The full classification report is saved in:

```txt
rf_results/classification_report.txt
```

### Figure 6. Confusion Matrix

![Confusion Matrix](figures/fig6_confusion_matrix.png)

The confusion matrix shows which classes are predicted correctly and which classes are commonly confused with each other.

---

## 11. Discussion

The Random Forest model can classify cropped traffic sign images after preprocessing.
The pipeline is simple and easy to understand because each step is clearly separated: cropping, resizing, normalization, flattening, training, and evaluation.

However, the model also has some limitations. Since the input image is flattened into a one-dimensional vector, the model does not preserve the spatial structure of the traffic sign very well. This means that shape, edge, and position information may not be learned as effectively as in image-specific models.

Misclassification may happen when:

* Traffic signs have similar shapes.
* Traffic signs have similar colors.
* The cropped image is blurry or too small.
* The bounding box does not crop the sign accurately.
* Some classes have fewer training samples than others.

Even with these limitations, Random Forest is still useful as a traditional machine learning baseline for cropped traffic sign classification.

---

## 12. Limitations

The main limitations of this Random Forest branch are:

* The model requires cropped traffic sign images as input.
* It cannot directly locate traffic signs in full road-scene images.
* Flattened pixel features do not preserve image spatial structure well.
* The model may be affected by blur, lighting changes, and poor crop quality.
* The current model uses fixed hyperparameters.
* Class labels are numeric, so a class mapping file is needed for better readability.

---

## 13. Future Improvements

Possible improvements include:

* Add a `classes.json` file to map numeric class IDs to readable traffic sign names.
* Add a prediction script for testing one new cropped image.
* Use better feature extraction methods such as HOG or color histogram.
* Tune hyperparameters using GridSearchCV or RandomizedSearchCV.
* Add a validation set for model selection.
* Analyze correct and incorrect prediction examples.
* Improve visualization of confusion matrix and experiment results.

---

## 14. Conclusion

This report presented the Random Forest branch for traffic sign classification.
The process starts from annotated images, crops the traffic sign regions, resizes and normalizes the cropped images, converts them into flattened feature vectors, and trains a Random Forest classifier.

The Random Forest model provides a simple and explainable traditional machine learning approach. Although it has limitations in image representation, it is useful for understanding how classical machine learning models perform on cropped traffic sign classification tasks.

---

## 15. Figures List

| Figure   | Description                                        |
| -------- | -------------------------------------------------- |
| Figure 1 | Random Forest preprocessing and training pipeline. |
| Figure 2 | Example original image with bounding box.          |
| Figure 3 | Example cropped traffic sign images.               |
| Figure 4 | Flattening process from image to feature vector.   |
| Figure 5 | Hyperparameter accuracy comparison chart.          |
| Figure 6 | Confusion matrix of the Random Forest model.       |

---

## 16. Files Used in This Branch

| File                        | Purpose                                                                |
| --------------------------- | ---------------------------------------------------------------------- |
| `crop_check.py`             | Checks whether bounding boxes are converted and cropped correctly.     |
| `create_rf_dataset.py`      | Crops traffic sign regions and creates the Random Forest dataset.      |
| `prepare_rf_features.py`    | Resizes, normalizes, and flattens cropped images into feature vectors. |
| `train_random_forest.py`    | Trains the Random Forest model and saves it.                           |
| `evaluate_random_forest.py` | Evaluates the trained model and saves result files.                    |
| `metadata/classes.json`     | Maps numeric class IDs to readable class names.                        |

---

## 17. References

This report is based on the Random Forest implementation and experiments conducted in this branch of the ML2 course project.
