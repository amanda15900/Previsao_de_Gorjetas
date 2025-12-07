import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os


def load_data(path="data/processed/tips_clean.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=["tip"])
    y = df["tip"]
    return X, y


def train_model(X, y, test_size=0.2, random_state=42):
    """Treina o modelo de regressão linear."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Calcula métricas de avaliação."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    return r2, rmse, y_pred


def save_model(model, path="models/linear_regression.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Modelo salvo em: {path}")


def load_model(path="models/linear_regression.pkl"):
    return joblib.load(path)
