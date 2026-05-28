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

## Dataset Structure

The original dataset should be organized as follows:

```txt
archive/
├── images/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
├── labels/
│   ├── image_1.txt
│   ├── image_2.txt
│   └── ...
└── split_dataset/
    ├── train_files.txt
    └── test_files.txt
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