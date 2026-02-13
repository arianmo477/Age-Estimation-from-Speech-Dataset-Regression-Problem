# Age Estimation from Speech using Machine Learning

## Overview

This project focuses on predicting a speaker’s age from speech recordings using supervised machine learning regression models. The system uses both acoustic and linguistic features extracted from audio signals and builds an end-to-end pipeline for preprocessing, training, and evaluation.

The objective is to accurately estimate age while maintaining good generalization performance. Model performance is evaluated using Root Mean Squared Error (RMSE).

---

## Dataset

The dataset is divided into two parts:

### Development Set
- Approximately 2,900 labeled samples
- Used for training and validation

### Evaluation Set
- Approximately 700 unlabeled samples
- Used for final predictions and submission

### Features

Acoustic features:
- pitch statistics (mean, max, min)
- jitter and shimmer
- energy
- tempo
- zero-crossing rate
- spectral centroid
- harmonics-to-noise ratio

Linguistic features:
- gender
- ethnicity
- number of words
- number of characters
- number of pauses
- silence duration

---

## Methodology

### Preprocessing
- Missing value imputation using median (numerical) and most frequent (categorical)
- Feature scaling using StandardScaler
- One-hot encoding for categorical variables
- Automated transformation using ColumnTransformer

### Handling Imbalance
- Sample weighting based on ethnicity distribution to reduce bias toward majority groups

### Models
Two ensemble regression models were implemented:

- Random Forest Regressor
- Gradient Boosting Regressor

### Hyperparameter Tuning
- GridSearchCV with 3-fold cross-validation
- Optimization based on negative mean squared error

---

## Results

Gradient Boosting achieved the best overall performance.

Typical results:

- Train RMSE: ~7
- Test RMSE: ~10

This indicates good generalization with only moderate overfitting.

---

## Project Structure

├── development.csv
├── evaluation.csv
├── main.py
├── environment.yml
├── README.md
└── submission.csv


---

## Installation

### 1. Install Conda (Miniconda recommended)

### 2. Create environment

conda env create -f environment.yml
conda activate age-estimation-speech


---

## Running the Project

### Run as script

### Run as notebook

---

## Output

The script generates:

- Best model selection
- Train/Test RMSE
- `submission.csv` containing predictions for the evaluation set

---

## Technologies Used

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- librosa

---

## Future Improvements

- XGBoost or LightGBM models
- Deep learning audio embeddings
- Additional feature engineering
- Larger hyperparameter search
- Data augmentation

---

## Author

Arian Mohammadi  
Politecnico di Torino  
Data Science and Engineering


