import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel

DATASET_PATH = "data/train"

def extract_features(image):
    features = {}

    # Resize image
    image = cv2.resize(image, (128, 128))

    # Color features: Mean and standard deviation in RGB
    for i, color in enumerate(["R", "G", "B"]):
        features[f"mean_{color}"] = np.mean(image[:, :, i])
        features[f"std_{color}"] = np.std(image[:, :, i])

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Statistical features: Mean and variance of grayscale intensity
    features["mean_gray"] = np.mean(gray)
    features["var_gray"] = np.var(gray)

    # Edge density using Sobel filter
    edges = sobel(gray)
    features["edge_density"] = np.sum(edges) / (128 * 128)

    # Texture features from GLCM
    glcm = graycomatrix(gray, [1], [0], symmetric=True, normed=True)
    features["contrast"] = graycoprops(glcm, "contrast")[0, 0]
    features["homogeneity"] = graycoprops(glcm, "homogeneity")[0, 0]

    return features

data = []
for category in os.listdir(DATASET_PATH):
    category_path = os.path.join(DATASET_PATH, category)
    if not os.path.isdir(category_path):
        continue

    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        features = extract_features(image)
        features["label"] = category
        data.append(features)

df = pd.DataFrame(data)
df.to_csv("features.csv", index=False)
print("Features saved successfully.")
