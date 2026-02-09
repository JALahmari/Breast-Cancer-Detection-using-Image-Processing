# Breast Cancer Detection using Image Processing and Random Forest

# Project Overview
This project aims to classify mammogram images into 'Normal' or 'Cancer' categories using image processing techniques for feature extraction combined with a Random Forest classifier. The goal is to demonstrate how feature engineering from medical images can significantly improve classification accuracy compared to using raw pixel data.

# Dataset
The project uses a subset of the MIAS (Mammographic Image Analysis Society) dataset, preprocessed and stored as NumPy arrays (.npy files) for training, validation, and testing. The labels are binarized, where 0 represents 'Normal' and 1 represents 'Cancer'.

# Expected Dataset Files:

MIAS_X_train_roi_multi.npy
MIAS_y_train_roi_multi.npy
MIAS_X_test_roi_multi.npy
MIAS_y_test_roi_multi.npy
MIAS_X_valid_roi_multi.npy
MIAS_y_valid_roi_multi.npy
Ensure these files are in the same directory as your notebook or provide the correct paths.

# Methodology
Data Loading and Binarization:

Loads image data (X) and labels (y) from .npy files.
Binarizes labels: 1 and 2 are mapped to 1 (Cancer), others to 0 (Normal).
Feature Extraction using Image Processing:

# For each image:
Converts to grayscale.
Applies Gaussian blur for noise reduction.
Performs histogram equalization for contrast enhancement.
Applies Otsu's thresholding to segment regions of interest.
Performs morphological closing to refine segmented regions.
Contour Analysis: Identifies contours and extracts features like:
suspicious_count: Number of contours with circularity below 0.70.
suspicious_area: Total area of suspicious contours.
avg_circularity, avg_solidity, avg_extent: Average shape descriptors for suspicious regions.
GLCM (Gray-Level Co-occurrence Matrix) Features: Extracts textural features from the equalized image:
contrast, homogeneity, energy.
# Model Training:

Baseline Model (Raw Data): A Random Forest Classifier is trained directly on flattened pixel data to serve as a baseline.
Processed Model (Extracted Features): A Random Forest Classifier is trained on the features extracted through image processing.
Evaluation:

Models are evaluated using test set accuracy, confusion matrices, and classification reports.
An interactive visualization allows inspecting individual validation samples, showing the original image, processing steps, detected contours, ground truth, and prediction.
Results
Baseline Accuracy (Raw Pixel Data): Approximately 64.58%
Improved Accuracy (Image Processing Features): Approximately 83.33%
The results demonstrate a significant improvement in classification performance (nearly 20% increase in accuracy) by leveraging domain-specific feature extraction through image processing techniques.

# How to Run
Prerequisites: Ensure you have Python installed, preferably Python 3.x.
Dependencies: Install the required libraries using pip:
pip install -r requirements.txt
Dataset: Place the .npy dataset files in the same directory as the notebook.
Execute Notebook: Run the Jupyter Notebook or Colab notebook cells sequentially.
The interactive visualization cell will prompt you for a start and end index to visualize validation samples.

# Libraries Used
NumPy
OpenCV (cv2)
Matplotlib
Scikit-learn
Scikit-image
