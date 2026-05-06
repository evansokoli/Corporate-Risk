import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
def main():
    df = pd.read_csv('Data/Cleaned/cleaned_data.csv')

    target = 'bankrupt'

    features = [
    'log_market_value',
    'net_income',
    'ebitda',
    'debt_ratio',
    'current_ratio',
    'net_profit_margin',
    'interest_rate_pressure'
    ]

    x = df[features]
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(x_train, y_train)

    y_probs = model.predict_proba(x_test)[:, 1]
    y_pred = (y_probs >= 0.1).astype(int)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\n Confusion Matrix\n", confusion_matrix(y_test, y_pred))
    print("\n Classification Report\n", classification_report(y_test, y_pred))

    joblib.dump(model, 'Models/random_forest_model.pkl')

if __name__ == "__main__":
    main()




