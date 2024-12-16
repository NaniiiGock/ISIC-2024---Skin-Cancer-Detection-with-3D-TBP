# ISIC-2024---Skin-Cancer-Detection-with-3D-TBP

## Table of Contents

- Project Overview
- Dataset
- Methodology
  - Tabular Data
  - Image Data
  - Ensemble Models
- Evaluation
- Results


## Project Overview

This repository contains the implementation for our Machine learning project, [*Skin Cancer Detection with 3D-TBP*](https://www.kaggle.com/competitions/isic-2024-challenge/overview), developed for the ISIC 2024 Challenge. The goal is to build machine learning models that can accurately distinguish between malignant and benign skin lesions using both image and metadata, even in low-quality smartphone-like photographs.

## Dataset

### Data Sources

- **ISIC 2024 Challenge Dataset**
  - Over 400,000 images of individual skin lesions.
  - Metadata with 54 features, including patient demographics, lesion characteristics, and diagnostic labels.
- **External Datasets**
  - [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
  - [PH² Dataset](https://www.kaggle.com/datasets/kliuiev/ph2databaseaddi)
  - [SIIM-ISIC 2020 Dataset](https://www.kaggle.com/c/siim-isic-melanoma-classification/overview)

### Data Distribution
- Malignant cases are significantly underrepresented.
<img src="https://github.com/NaniiiGock/ISIC-2024---Skin-Cancer-Detection-with-3D-TBP/blob/main/results/target_value_distribution.png" alt="target value distribution" width="600">

## Methodology

### Tabular Data

- [**Preprocessing Techniques**](https://github.com/NaniiiGock/ISIC-2024---Skin-Cancer-Detection-with-3D-TBP/blob/main/src/XGBoost_Tabular.ipynb)
  - Handling Missing Values: Imputation (e.g., mean filling) for numerical features.
  - Categorical Features: One-hot encoding for variables like sex and lesion location.
  - Normalization: Scaling features to ensure balanced contributions.
  - Feature Selection: Correlation matrix and PCA for dimensionality reduction.
  - Balancing Classes: Applied **SMOTE** and **ADASYN** to oversample malignant cases.
- [**Models Explored**](https://github.com/NaniiiGock/ISIC-2024---Skin-Cancer-Detection-with-3D-TBP/blob/main/src/XGBoost_Tabular.ipynb)
  - Random Forest, Extra Trees, XGBoost, and LightGBM.
  - LightGBM and XGBoost achieved the best results.
- [**Optimization**](https://github.com/NaniiiGock/ISIC-2024---Skin-Cancer-Detection-with-3D-TBP/blob/main/src/Xgboost%20and%20LGBM%20model%20fine%20tuning.ipynb)
  - Cross-validation (5-fold stratified).
  - Bayesian optimization for hyperparameter tuning.

### Image Data

- **Preprocessing Techniques**

  - Hair Removal: Used the [**DullRazor algorithm**](https://github.com/BlueDokk/Dullrazor-algorithm) to remove hair artifacts.
  - Image Resizing: All images resized to 224x224 pixels to ensure uniform input.
  - Data Augmentation: 1. Random horizontal and vertical flips. 2. Random resized cropping.
  - Normalization: Applied mean and standard deviation values to match pre-trained model requirements.

- **Models Explored**

  - MobileNet
  - [Vision Transformer (ViT)](https://github.com/NaniiiGock/ISIC-2024---Skin-Cancer-Detection-with-3D-TBP/blob/main/src/vit_final_version.py)

- **Cross-validation**

  Incorporated stratified cross-validation to improve robustness and minimize overfitting.



## Ensemble Models

Combined predictions from tabular and image models for better overall performance.

### Method

- Arithmetic Mean
- Geometric Mean
- Soft Voting
- Stacking
- Bagging

## Evaluation

- **Metric**: Partial Area Under the ROC Curve (pAUC) above an 80% True Positive Rate (TPR). Hence, scores range from [0.0, 0.2].

- **Implementation**: [ISIC pAUC above TPR](https://www.kaggle.com/code/metric/isic-pauc-abovetpr)

## Results

### Tabular Models
- Best model: XGBoost
- Results on Kaggle Submission
<img src="https://github.com/NaniiiGock/ISIC-2024---Skin-Cancer-Detection-with-3D-TBP/blob/main/results/Tabular%20Model%20Comparison.png" alt="Tabular Model Comparison" width="600">

### Image Models
- ViT Training Result
<img src="https://github.com/NaniiiGock/ISIC-2024---Skin-Cancer-Detection-with-3D-TBP/blob/main/results/vit_model_training_results.png" alt="ViT model training result" width="1000">
- MobileNet

### Ensemble Models




