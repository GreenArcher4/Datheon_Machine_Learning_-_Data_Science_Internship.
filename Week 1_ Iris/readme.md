# Multi-Class Classification Benchmark: Iris Dataset

## Overview
This project builds a classification pipeline that predicts the
species of an iris flower from its measurements. Two models are trained and
benchmarked on accuracy, precision, recall, and F1-score.
Completed as Task 1 of the Datatheon internship program.

## Goals
- Load, inspect, and explore the Iris dataset
- Prepare the data (train/test split, feature scaling)
- Train and compare Logistic Regression and Decision Tree
- Evaluate with accuracy, a classification report, and confusion matrices
- Identify which model performs best

## Tools
Python, Pandas, Scikit-Learn, Seaborn, Matplotlib

## Data Dictionary
Dataset: `Iris Dataset.csv` | 150 rows | 3 species, 50 flowers each

| Column | Type | Description |
|---|---|---|
| sepal_length | float | Sepal length in cm |
| sepal_width | float | Sepal width in cm |
| petal_length | float | Petal length in cm |
| petal_width | float | Petal width in cm |
| species | categorical | Target: setosa, versicolor, or virginica |

## Methodology

### 1. Load the data
Loaded with `Iris Dataset.csv`. Note: the file was UTF-16 encoded, so
`encoding="utf-16"` was needed.

### 1.2. Data inspection and preprocessing
* Used `.info(), .shape(), .describe(),.isnull().sum(), .columns(), .duplicated().sum()`.
* I found no null values but I found a duplicate. I used '.drop_duplicates(inplace=True)' to drop it

### 2. Perform exploratory pairplots with Seaborn
Pairplot colored by species. Setosa forms a separate cluster; versicolor and
virginica overlap, so they are the hardest to separate. Petal measurements
separate the species best.

### 3. Split features and target and feature scaling using StandardScaler()
X = four measurements, y = species. 80/20 split with `random_state=42` and
`stratify=y`, giving 120 training and 30 test samples.
Applied `StandardScaler`, fit on the training set only and applied to the
test set, to avoid data leakage.

### 4. Model training
Two models were trained on the scaled training data:
Logistic Regression, Decision Tree.

### 5. Prediction and evaluation
Each model predicted on the held-out test set and was scored with accuracy,
precision, recall, F1, and a confusion matrix.

## Results
K-Nearest Neighbors achieved the highest test accuracy (96.7%), followed by Logistic Regression and Decision Tree (93.3%) and Random Forest (90.0%).
| Model | Accuracy | 
|---|---|
| Logistic Regression | [0.933 ] |
| Decision Tree | [ 0.933] |



