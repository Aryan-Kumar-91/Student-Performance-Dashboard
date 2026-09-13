"""
Module: Linear Regression & Regression Diagnostics
Subject: Statistical Machine Learning for Data Science

Fits Ordinary Least Squares (OLS) Linear Regression for Final Academic Score prediction,
computes goodness-of-fit metrics (R², Adj R², MAE, RMSE), statistical significance of coefficients,
and runs regression diagnostics (residual analysis, Q-Q plots, homoscedasticity assessment).
"""

import os
import joblib
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import statsmodels.api as sm
from scipy import stats

from src.data_preprocessing import prepare_modeling_data

def train_linear_regression(df: pd.DataFrame, random_state: int = 42) -> Dict[str, Any]:
    """
    Trains OLS Linear Regression and generates statistical inference metrics.
    """
    prep = prepare_modeling_data(df, target_type="regression", random_state=random_state)
    
    X_train = prep["X_train"]
    X_test = prep["X_test"]
    y_train = prep["y_train"]
    y_test = prep["y_test"]
    
    # 1. Scikit-Learn Linear Regression Model
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, y_train)
    
    y_train_pred = sklearn_model.predict(X_train)
    y_test_pred = sklearn_model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    n_test = len(y_test)
    p_features = X_test.shape[1]
    adj_r2 = 1.0 - ((1.0 - test_r2) * (n_test - 1) / (n_test - p_features - 1))
    
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = root_mean_squared_error(y_test, y_test_pred)
    
    # Residuals
    residuals_test = y_test.values - y_test_pred
    residual_mean = float(np.mean(residuals_test))
    residual_std = float(np.std(residuals_test, ddof=1))
    
    # 2. Statsmodels OLS for Comprehensive Statistical Significance
    X_train_sm = sm.add_constant(X_train)
    sm_model = sm.OLS(y_train, X_train_sm).fit()
    
    # Extract Coefficient Summary Table
    coef_df = pd.DataFrame({
        "Feature": ["Intercept"] + list(X_train.columns),
        "Coefficient (β)": sm_model.params.values.round(4),
        "Std Error": sm_model.bse.values.round(4),
        "t-Statistic": sm_model.tvalues.values.round(3),
        "p-Value": sm_model.pvalues.values.round(4),
        "95% CI Lower": sm_model.conf_int()[0].values.round(4),
        "95% CI Upper": sm_model.conf_int()[1].values.round(4)
    })
    
    # Assumption Checks
    # a. Normality of residuals (Shapiro-Wilk)
    _, res_norm_p = stats.shapiro(residuals_test)
    residuals_normal = bool(res_norm_p >= 0.05)
    
    # b. Homoscedasticity (Breusch-Pagan / Goldfeld-Quandt / Correlation of abs residuals vs fitted)
    _, bp_p, _, _ = sm.stats.diagnostic.het_breuschpagan(sm_model.resid, X_train_sm)
    homoscedastic = bool(bp_p >= 0.05)
    
    # Save Model Artifact
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_artifact_path = os.path.join(models_dir, "regression_model.pkl")
    
    artifact = {
        "model": sklearn_model,
        "preprocessor": prep["preprocessor"],
        "raw_features": prep["raw_features"],
        "transformed_features": prep["transformed_features"],
        "metrics": {
            "test_r2": test_r2,
            "adj_r2": adj_r2,
            "mae": mae,
            "rmse": rmse
        }
    }
    joblib.dump(artifact, model_artifact_path)
    
    return {
        "model": sklearn_model,
        "sm_model": sm_model,
        "prep": prep,
        "y_test": y_test.values,
        "y_test_pred": y_test_pred,
        "residuals_test": residuals_test,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "adj_r2": adj_r2,
        "mae": mae,
        "rmse": rmse,
        "coef_df": coef_df,
        "residuals_normal": residuals_normal,
        "res_norm_p": float(res_norm_p),
        "homoscedastic": homoscedastic,
        "bp_p": float(bp_p),
        "artifact_path": model_artifact_path
    }
