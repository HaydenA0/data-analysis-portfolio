import joblib
import pandas as pd
from preprocess import clean, numerization, split_data
from plotting import (
    plot,
    plot_xgboost_tree,
    analyze_logistic_regression,
    plot_analyze_decision_tree,
    plot_model_performance_comparison,
)
from train import (
    train_xgb_model,
    train_logistic_regression,
    train_decision_tree,
)

from evaluation import evaluate_model


raw_data = pd.read_csv("Project/data/raw.csv")

plot(raw_data)
processed_data = numerization(clean(raw_data))

X_train, X_val, y_train, y_val = split_data(processed_data)

xgb_model = train_xgb_model(X_train, y_train, 4, 4)
log_model = train_logistic_regression(X_train, y_train)
tree_model = train_decision_tree(X_train, y_train)


joblib.dump(xgb_model, "Project/models/xgb_model.pkl")

joblib.dump(log_model, "Project/models/log_model.pkl")

joblib.dump(tree_model, "Project/models/tree_model.pkl")

plot_xgboost_tree(xgb_model)

analyze_logistic_regression(log_model)

plot_analyze_decision_tree(tree_model)

evaluate_model(xgb_model, X_train, X_val, y_train, y_val)
evaluate_model(log_model, X_train, X_val, y_train, y_val)
evaluate_model(tree_model, X_train, X_val, y_train, y_val)


plot_model_performance_comparison()
