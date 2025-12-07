# src/model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.dummy import DummyRegressor

def load_data(path="data/processed/tips_clean.csv"):
    """Carrega X e y do CSV processado."""
    df = pd.read_csv(path)
    X = df.drop(columns=["tip"])
    y = df["tip"]
    return X, y

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Treina modelo de Regressão Linear. Retorna (model, X_test, y_test, X_train, y_train)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test, X_train, y_train

def evaluate_model(model, X_test, y_test):
    """Avalia um modelo retornando r2, rmse e y_pred."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return r2, rmse, y_pred

def baseline_evaluate(X, y, test_size=0.2, random_state=42):
    """
    Cria e avalia um baseline simples (DummyRegressor com strategy='mean').
    Retorna r2, rmse.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return r2, rmse

def cross_validate_model(estimator, X, y, cv=5, scoring="neg_root_mean_squared_error"):
    """
    Executa cross-validation e retorna dict com médias e desvios das métricas.
    Note: sklearn usa scores negativos para losses — aqui é compatível com RMSE usando 'neg_root_mean_squared_error'.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    # R2:
    r2_scores = cross_val_score(estimator, X, y, cv=kf, scoring="r2")
    # RMSE (neg root mse)
    rmse_scores_neg = cross_val_score(estimator, X, y, cv=kf, scoring=scoring)
    rmse_scores = -rmse_scores_neg  # tornar positivo

    results = {
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "n_folds": cv
    }
    return results

def save_model(model, path="models/linear_regression.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Modelo salvo em: {path}")

def load_model(path="models/linear_regression.pkl"):
    return joblib.load(path)
