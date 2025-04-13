from preprocess import clean, numerization
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# Load the raw data
raw = pd.read_csv("Project/data/raw.csv")


cleaned = clean(raw)
processed = numerization(cleaned)

processed.to_csv("Project/data/processed_data.csv", index=False)


def train_xgb_model(X, y, n_estimators, max_depth):
    model = XGBRegressor(
        random_state=42, n_jobs=-1, n_estimators=n_estimators, max_depth=max_depth
    )
    model.fit(X, y)
    return model


def train_logistic_regression(X, y):
    model = LogisticRegression(max_iter=10_000, n_jobs=-1)
    model.fit(X, y)
    return model


def naive_model(X_input):
    return np.zeros(len(X_input))


def train_decision_tree(X, y):
    model = DecisionTreeClassifier(random_state=42, max_depth=3)
    model.fit(X, y)
    return model
