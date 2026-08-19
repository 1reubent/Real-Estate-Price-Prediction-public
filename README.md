# Real Estate Price Prediction

## Overview

This project predicts residential sale prices using the **Ames, Iowa Housing dataset** (79 features, ~2,900 records). Three regression models — **Linear**, **Ridge**, and **Lasso** — are trained and compared, each on both the raw dataset and a feature-engineered version built from 15+ new features encoding real estate domain knowledge (e.g. total square footage, home age, neighborhood price levels).

The target variable (`SalePrice`) is log-transformed to correct for right skew before modeling. **Ridge regression on the feature-engineered dataset performed best**, achieving **R² = 0.937** and **RMSE = 0.108** on the log-transformed target, showing that regularization combined with domain-informed feature engineering meaningfully improves prediction accuracy over a plain linear baseline.

## Key Results

| Model                     | RMSE (Test) | MAE (Test) | R² (Test) | CV RMSE |
|---------------------------|:-----------:|:----------:|:---------:|:-------:|
| Linear Regression         |   0.1330    |   0.0905   |  0.9043   | 0.1582  |
| Ridge Regression          |   0.1182    |   0.0798   |  0.9245   | 0.1407  |
| Lasso Regression          |   0.1202    |   0.0807   |  0.9220   | 0.1435  |
| Linear Regression (FE)    |   0.1161    |   0.0830   |  0.9270   | 0.1418  |
| **Ridge Regression (FE)** | **0.1076**  | **0.0749** | **0.9374**| **0.1280**|
| Lasso Regression (FE)     |   0.1080    |   0.0755   |  0.9370   | 0.1269  |

*FE = trained on the feature-engineered dataset. Metrics are computed on the log-transformed sale price.*

The most predictive features were `TotalSF_Log` (total square footage), `OverallQual`, `GrLivArea_Log`, and `Neighborhood_Price` (an engineered feature encoding mean sale price by neighborhood).

## Dataset

The [Ames Housing dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset) (`AmesHousing.csv`) contains records of residential property sales in Ames, Iowa from 2006–2010, with 79 explanatory variables covering:
- Property characteristics (square footage, rooms, bathrooms)
- Location (neighborhood, zoning)
- Quality/condition ratings
- Time-related features (year built, remodeled, sold)
- Special features (fireplaces, pools, garages)

## Methodology

The full analysis is implemented in [final_project_code.ipynb](final_project_code.ipynb):

1. **Exploratory Data Analysis** — feature distributions, skewness, correlation analysis, and categorical feature trends
2. **Feature Engineering** — missing value imputation, 15+ new features (e.g. `TotalSF`, `HouseAge`, `TotalBath`, `LotRatio`, `Neighborhood_Price`), ordinal encoding of quality features, log-transforms of skewed numeric features, and one-hot encoding of categorical features
3. **Preprocessing** — separate pipelines built for the raw and feature-engineered datasets
4. **Model Training** — Linear, Ridge, and Lasso regression, each trained on both datasets (6 models total), with hyperparameter tuning via `GridSearchCV`
5. **Evaluation** — RMSE, MAE, R², and 5-fold cross-validation on the test set
6. **Prediction Analysis** — actual vs. predicted plots and residual diagnostics for the best model
7. **Feature Importance** — comparison of model coefficients to identify the strongest price drivers

## Repository Structure

```
.
├── AmesHousing.csv          # Raw dataset
├── final_project_code.ipynb # Full analysis: EDA, feature engineering, modeling, evaluation
└── report/
    ├── report.tex           # Written report (NeurIPS-style)
    └── report.pdf           # Compiled report
```

## Getting Started

**Requirements:** Python 3, Jupyter, and the following packages:

```
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

**Run the analysis:**

```
jupyter notebook final_project_code.ipynb
```

## Report

A full write-up of the methodology, insights, and results is available in [report/report.pdf](report/report.pdf).

## Authors

Reuben Thomas, Miguel Nino Adalla, Nicole Krivokapic
