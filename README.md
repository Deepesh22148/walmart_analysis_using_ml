# Walmart Customer Behavior and Purchasing Patterns Analysis

## Project Overview
This project aims to analyze customer purchasing behavior at Walmart using a combination of supervised and unsupervised machine learning techniques. The primary objectives are:
1. **Predict purchase amounts** based on demographic features.
2. **Classify product categories** that customers are likely to purchase.

### Key Features
- **Regression Models** for predicting purchase amounts:
  - Decision Trees
  - Random Forests
  - Gradient Boosting
- **Classification Models** for product category prediction:
  - k-Means Clustering for customer segmentation
  - k-Nearest Neighbors (k-NN) for classification

### Contributors
- Akansh Gupta
- Chawan Srujeeth
- Deepesh Sahu
- Vinay Kumar Dubey

---

## Table of Contents
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Methodology](#methodology)
- [Results](#results)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [References](#references)

---

## Dataset
The dataset used for this project is provided by Walmart and contains information on:
- **Customer Demographics:** Age, Gender, Occupation, etc.
- **Transaction Data:** Purchase Amount, Product Categories, etc.

### Data Preprocessing
1. **Encoding categorical data:** Binary encoding for gender, marital status, and city categories.
2. **Outlier Removal:** Outliers in purchase amounts were removed to improve model performance.

---

## Methodology

### 1. Predicting Purchase Amounts
- **Decision Trees:** Captured feature interactions but were prone to overfitting.
- **Random Forests:** Improved generalization through ensemble learning.
- **Gradient Boosting:** Achieved the best performance by minimizing prediction errors.

### 2. Classifying Product Categories
- **k-Means Clustering:** Segmented customers based on demographic and purchasing behavior.
- **k-Nearest Neighbors (k-NN):** Used enriched features from clustering for classification.

---

## Results
### Regression Performance
- **Explained Variance:** 64.2% on the test data.
- **Best Model:** Gradient Boosting with consistent performance across K-fold cross-validation.

### Classification Performance
- **k-Means Clustering Accuracy:** 73%
- **k-NN Accuracy:** 34% (Challenges due to class imbalance and overlapping feature distributions).

---

## Setup and Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/walmart-customer-analysis.git
