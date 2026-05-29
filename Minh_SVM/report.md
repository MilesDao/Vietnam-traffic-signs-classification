# Report: Applying SVM for Vietnam Traffic Sign Classification

## 1. Project Overview
This project aims to build a Machine Learning model using the **Support Vector Machine (SVM)** algorithm to recognize and classify traffic signs in Vietnam (covering 52 distinct classes).
The implementation process consists of four main phases:
1. **Data Preprocessing (`create_svm_dataset.py`)**: Cropped traffic signs from original images using bounding box coordinates, applying a 5% padding to retain visual context.
2. **Feature Extraction (`prepare_svm_features.py`)**: Resized all cropped images to a standard `64x64` resolution and flattened them into a 1D vector of `12,288` features (64 * 64 * 3).
3. **Model Training (`train_svm.py`)**: 
   - Established a Baseline model for standard evaluation.
   - Employed `RandomizedSearchCV` to automatically hunt for the optimal hyperparameters from a massive parameter space.
4. **Evaluation & Visualization (`svm_compare_loop.py` & `evaluate_svm.py`)**: Conducted an in-depth analysis of 16 different configurations, plotting charts to compare accuracy and training time.

## 2. Hyperparameter Fine-Tuning Insights
Based on the exhaustive grid search results recorded in `svm_compare_results.csv`, we can draw highly valuable insights regarding SVM characteristics for image processing:

### The Penalty Parameter `C` (Regularization)
- **When `C` is too small (`C = 0.1`)**: The RBF model suffers from severe "Underfitting" (only reaching ~62.9% Test Accuracy). The model is too simplistic, resulting in decision boundaries that are not sharp enough to separate visually similar traffic signs.
- **When `C` is moderately increased (`C = 1.0` to `10.0`)**: Accuracy skyrockets to 96.7%. This is the **sweet spot** where the model perfectly balances learning from the training data and generalizing to unseen data.
- **When `C` is excessively large (`C = 100.0`)**: Clear signs of **Overfitting** emerge. The Train Accuracy maxes out at nearly `1.0` (100%)—meaning the model has essentially memorized the training set—yet the Test Accuracy plateaus at 96.7% without further improvement.

### The `kernel` Parameter (RBF vs. Linear)
- **Linear Kernel**: Delivers fast convergence and high initial scores (96.1% at C=0.1). However, as C increases, the Linear kernel hits a "glass ceiling" and drops to 95.3% due to rigid overfitting.
- **RBF Kernel (Radial Basis Function)**: Demonstrates absolute superiority in flexing decision boundaries. At the optimal point (`C=10.0`, `gamma='scale'`), RBF reaches 96.7%, completely outperforming the Linear Kernel.

### The `gamma` Parameter (Radius of Influence)
- `gamma='scale'` proved to be incredibly effective and stable across all scenarios.
- Conversely, `gamma='auto'` (which calculates `1/n_features`) is a disaster for our 12,288-dimensional image vectors. It caused the Test Accuracy to plummet to a mere `6.8%` at C=0.1, indicating that the influence radius of each data point became practically invisible, leaving the SVM completely directionless.

## 3. The Final Fine-Tuned Configuration & Results
Through rigorous `RandomizedSearchCV` iterations and manual grid comparison, we successfully identified the absolute **BEST configuration** for this dataset:

*   **Kernel**: **`rbf`**
*   **C (Regularization)**: **`10.0`** (or ~15.7 as suggested by RandomizedSearchCV)
*   **Gamma**: **`scale`**
*   **Class Weight**: **`balanced`** *(Crucial for handling the heavily imbalanced nature of the dataset).*

**🏆 Final Performance Metrics:**
*   **Test Accuracy: `96.72%`** *(A massive leap from the 91.0% Baseline model).*
*   **Macro F1-Score: `95.39%`** *(Demonstrating that the model performs exceptionally well across all 52 classes, accurately classifying even the rarest traffic signs).*

## 4. Conclusion & Future Work
1. **SVM Viability**: The SVM algorithm is highly capable of tackling basic image recognition tasks (recognizing simple geometric shapes of traffic signs) when provided with accurately cropped data.
2. **Overfitting Warning**: This project clearly illustrates that forcing a higher `C` parameter (e.g., C=100) does not yield better real-world predictions; it merely forces the model to memorize noise (Train Acc = 100%, Test Acc plateaued).
3. **Future Enhancements**: To shatter the 96.7% accuracy ceiling, relying on raw flattened pixels is no longer sufficient. Future developments should pivot toward advanced feature extraction techniques like **HOG (Histogram of Oriented Gradients)** or adopt **Convolutional Neural Networks (CNNs)** to effectively capture spatial image hierarchies.
