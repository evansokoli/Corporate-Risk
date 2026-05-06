# CS210 Project

## Overview
This project aims to predict corporate bankruptcy using financial and macroeconomic data. 
By applying machine learning models such as Random Forest, the goal is to identify patterns 
that distinguish bankrupt firms from non-bankrupt ones.

## Dataset
The dataset contains:
- Company financial statement information - Kaggle
- Macroeconomic indicators - Federal Reserve Economic Data (FRED)

## Features
- Debt Ratio
- Current Ratio
- EBITDA
- Net Income
- Market Capitalization
- Liquidity Measures

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- SQLite

## Machine Learning Models
### Random Forest Classifier
The primary model used in this project is a Random Forest classifier.

Additional work included:
- Feature engineering
- Threshold tuning
- Handling class imbalance
- Feature importance analysis

## Results
Model performance was evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

Example findings:
- Debt-related features were strong bankruptcy predictors
- Feature engineering improved recall performance
- Threshold tuning helped reduce false negatives


## Installation

### Clone Repository
```bash
git clone https://github.com/evansokoli/Corporate-Risk
cd CS210_Project
```

## Author
Evan Sokoli
