import os
import cv2
import joblib
import numpy as np

img_size = 64

model_path = "rf_models/random_forest_flatten.pkl"
img_path = "demo.jpg"

if not os.path.exists(model_path):
    print("Model not found:", model_path)
    exit()

if not os.path.exists(img_path):
    print("Image not found:", img_path)
    print("Please put a cropped traffic sign image named demo.jpg in the project folder.")
    exit()

rf = joblib.load(model_path)

img = cv2.imread(img_path)

if img is None:
    print("Cannot read image:", img_path)
    print("Please check if the image is valid or corrupted.")
    exit()

img = cv2.resize(img, (img_size, img_size))
img = img / 255.0

x = img.flatten().reshape(1, -1)

pred = rf.predict(x)[0]
prob = rf.predict_proba(x).max()

print("Predicted class:", pred)
print("Confidence:", round(prob * 100, 2), "%")