"""
Module: Logistic Regression & Imbalanced Classification
Subject: Statistical Machine Learning for Data Science

Fits Logistic Regression (GLM Logit Link) to predict student At-Risk status.
Evaluates baseline vs balanced (class_weight='balanced') classifiers on imbalanced academic data,
computes Odds Ratios, Confusion Matrices, Precision, Recall, F1-Score, and ROC-AUC curves.
"""

import os
import joblib
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
import statsmodels.api as sm

from src.data_preprocessing import prepare_modeling_data

def train_classification_models(df: pd.DataFrame, random_state: int = 42) -> Dict[str, Any]:
    """
    Trains Baseline Logistic Regression and Imbalance-Aware Logistic Regression.
    Strictly excludes Final_Score from predictor features to prevent target leakage.
    """
    prep = prepare_modeling_data(df, target_type="classification", random_state=random_state)
    
    X_train = prep["X_train"]
    X_test = prep["X_test"]
    y_train = prep["y_train"]
    y_test = prep["y_test"]
    
    # 1. Baseline Logistic Regression (Standard Maximum Likelihood)
    clf_baseline = LogisticRegression(class_weight=None, random_state=random_state, max_iter=1000)
    clf_baseline.fit(X_train, y_train)
    
    y_test_pred_base = clf_baseline.predict(X_test)
    y_test_prob_base = clf_baseline.predict_proba(X_test)[:, 1]
    
    # 2. Imbalance-Aware Logistic Regression (Cost-Sensitive / Balanced Weights)
    clf_balanced = LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=1000)
    clf_balanced.fit(X_train, y_train)
    
    y_test_pred_bal = clf_balanced.predict(X_test)
    y_test_prob_bal = clf_balanced.predict_proba(X_test)[:, 1]
    
    # Compute Metrics Function
    def get_metrics(y_true, y_pred, y_prob):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "specificity": specificity,
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_prob),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        }
        
    metrics_baseline = get_metrics(y_test, y_test_pred_base, y_test_prob_base)
    metrics_balanced = get_metrics(y_test, y_test_pred_bal, y_test_prob_bal)
    
    # Odds Ratios and Coefficients (from Balanced Model)
    feature_names = list(X_train.columns)
    coefs = clf_balanced.coef_[0]
    odds_ratios = np.exp(coefs)
    
    odds_df = pd.DataFrame({
        "Feature": feature_names,
        "Log-Odds Coef (β)": coefs.round(4),
        "Odds Ratio (exp(β))": odds_ratios.round(4)
    })
    
    # Add plain-English effect descriptions
    def interpret_or(row):
        or_val = row["Odds Ratio (exp(β))"]
        if or_val > 1.0:
            pct = (or_val - 1.0) * 100
            return f"+{pct:.1f}% higher odds of At-Risk per 1-SD increase"
        else:
            pct = (1.0 - or_val) * 100
            return f"-{pct:.1f}% lower odds of At-Risk per 1-SD increase"
            
    odds_df["Statistical Interpretation"] = odds_df.apply(interpret_or, axis=1)
    odds_df = odds_df.sort_values(by="Odds Ratio (exp(β))", ascending=False).reset_index(drop=True)
    
    # ROC Curves
    fpr_base, tpr_base, _ = roc_curve(y_test, y_test_prob_base)
    fpr_bal, tpr_bal, _ = roc_curve(y_test, y_test_prob_bal)
    
    # Precision-Recall Curves
    pr_prec_base, pr_rec_base, _ = precision_recall_curve(y_test, y_test_prob_base)
    pr_prec_bal, pr_rec_bal, _ = precision_recall_curve(y_test, y_test_prob_bal)
    
    # Save Model Artifact (Saving Balanced model as primary production inference model)
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_artifact_path = os.path.join(models_dir, "classification_model.pkl")
    
    artifact = {
        "model": clf_balanced,
        "preprocessor": prep["preprocessor"],
        "raw_features": prep["raw_features"],
        "transformed_features": prep["transformed_features"],
        "metrics_balanced": metrics_balanced,
        "metrics_baseline": metrics_baseline
    }
    joblib.dump(artifact, model_artifact_path)
    
    # Class breakdown in test set
    test_class_counts = pd.Series(y_test).value_counts().to_dict()
    
    return {
        "clf_baseline": clf_baseline,
        "clf_balanced": clf_balanced,
        "prep": prep,
        "y_test": y_test.values,
        "y_test_pred_base": y_test_pred_base,
        "y_test_prob_base": y_test_prob_base,
        "y_test_pred_bal": y_test_pred_bal,
        "y_test_prob_bal": y_test_prob_bal,
        "metrics_baseline": metrics_baseline,
        "metrics_balanced": metrics_balanced,
        "odds_df": odds_df,
        "roc_base": (fpr_base, tpr_base),
        "roc_bal": (fpr_bal, tpr_bal),
        "pr_base": (pr_prec_base, pr_rec_base),
        "pr_bal": (pr_prec_bal, pr_rec_bal),
        "test_class_counts": test_class_counts,
        "artifact_path": model_artifact_path
    }
