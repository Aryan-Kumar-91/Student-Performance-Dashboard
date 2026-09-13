"""
Module: Data Preprocessing
Subject: Statistical Machine Learning for Data Science

Handles dataset loading, verification, missing/duplicate audits, feature identification,
train-test splitting, and pipeline transformations with strict data leakage prevention.
"""

import os
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Loads student performance dataset from CSV.
    """
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "student_data.csv")
    
    if not os.path.exists(filepath):
        # Auto-generate if missing
        from data.generate_data import generate_student_dataset
        df = generate_student_dataset()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        return df
    
    return pd.read_csv(filepath)

def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs initial data audit (dimensions, missing values, duplicates, datatypes).
    """
    n_rows, n_cols = df.shape
    missing_counts = df.isnull().sum()
    duplicate_count = df.duplicated().sum()
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Exclude ID from analysis features
    if "Student_ID" in cat_cols:
        cat_cols.remove("Student_ID")
        
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "missing_counts": missing_counts,
        "total_missing": missing_counts.sum(),
        "duplicate_count": duplicate_count,
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols
    }

def prepare_modeling_data(
    df: pd.DataFrame, 
    target_type: str = "regression", 
    test_size: float = 0.20, 
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Prepares train/test splits and fitted ColumnTransformers.
    
    CRITICAL RULE:
    For classification (predicting At_Risk), Final_Score is excluded from X
    to prevent target leakage.
    """
    # Exclude IDs and Targets from feature matrix
    if target_type == "regression":
        y = df["Final_Score"].copy()
        feature_cols = [c for c in df.columns if c not in ["Student_ID", "Final_Score", "At_Risk"]]
    elif target_type == "classification":
        y = df["At_Risk"].copy()
        # Strictly exclude Final_Score to prevent target leakage
        feature_cols = [c for c in df.columns if c not in ["Student_ID", "Final_Score", "At_Risk"]]
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
        
    X = df[feature_cols].copy()
    
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Train-Test Split (stratified for classification to maintain class imbalance ratio)
    stratify_arg = y if target_type == "classification" else None
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )
    
    # Column Transformer: Standardize numeric features, One-Hot Encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_features)
        ]
    )
    
    # Fit ONLY on training data to prevent data leakage
    X_train_trans = preprocessor.fit_transform(X_train_raw)
    X_test_trans = preprocessor.transform(X_test_raw)
    
    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_features))
    transformed_feature_names = num_features + encoded_cat_names
    
    X_train_df = pd.DataFrame(X_train_trans, columns=transformed_feature_names, index=X_train_raw.index)
    X_test_df = pd.DataFrame(X_test_trans, columns=transformed_feature_names, index=X_test_raw.index)
    
    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train": X_train_df,
        "X_test": X_test_df,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "raw_features": feature_cols,
        "num_features": num_features,
        "cat_features": cat_features,
        "transformed_features": transformed_feature_names
    }
