"""
Module: Exploratory Data Analysis (EDA)
Subject: Statistical Machine Learning for Data Science

Calculates descriptive statistics, univariate and bivariate measures,
correlation structures, and automated observations.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats

def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rigorous descriptive statistics for all numerical features.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "At_Risk" in num_cols:
        num_cols.remove("At_Risk")
        
    stats_list = []
    for col in num_cols:
        series = df[col].dropna()
        stats_list.append({
            "Feature": col,
            "Count": int(series.count()),
            "Mean": round(series.mean(), 2),
            "Std Dev": round(series.std(), 2),
            "Median (Q2)": round(series.median(), 2),
            "IQR": round(series.quantile(0.75) - series.quantile(0.25), 2),
            "Min": round(series.min(), 2),
            "25% (Q1)": round(series.quantile(0.25), 2),
            "75% (Q3)": round(series.quantile(0.75), 2),
            "Max": round(series.max(), 2),
            "Skewness": round(stats.skew(series), 3),
            "Kurtosis": round(stats.kurtosis(series), 3)
        })
    return pd.DataFrame(stats_list)

def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Pearson correlation matrix for numerical features.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[num_cols].corr().round(3)

def generate_automated_insights(df: pd.DataFrame) -> List[str]:
    """
    Generates dynamic statistical observations from the dataset
    strictly avoiding causation claims (correlation != causation).
    """
    insights = []
    
    # 1. Target Correlation with Predictors
    corr = df.select_dtypes(include=[np.number]).corr()["Final_Score"].drop(["Final_Score", "At_Risk"], errors="ignore")
    top_pos = corr.idxmax()
    top_pos_val = corr.max()
    top_neg = corr.idxmin()
    top_neg_val = corr.min()
    
    insights.append(
        f"**Strongest Positive Linear Association:** `{top_pos}` exhibits the highest positive Pearson correlation with Final Score (r = {top_pos_val:.2f}), indicating that higher values of this metric statistically co-occur with higher academic scores."
    )
    
    insights.append(
        f"**Strongest Inverse Association:** `{top_neg}` has the strongest negative correlation with Final Score (r = {top_neg_val:.2f}), showing an inverse linear trend with student performance."
    )
    
    # 2. At-Risk Class Distribution
    risk_counts = df["At_Risk"].value_counts()
    risk_pct = df["At_Risk"].value_counts(normalize=True) * 100
    insights.append(
        f"**Class Distribution & Imbalance:** Out of {len(df)} students, {risk_counts.get(1, 0)} ({risk_pct.get(1, 0.0):.1f}%) are categorized as *At-Risk* (Final Score < 55) and {risk_counts.get(0, 0)} ({risk_pct.get(0, 0.0):.1f}%) are *Not At-Risk*. This creates a {risk_pct.get(0, 0.0):.0f}:{risk_pct.get(1, 0.0):.0f} class imbalance."
    )
    
    # 3. Attendance vs Risk Grouping
    mean_att_risk = df[df["At_Risk"] == 1]["Attendance_Rate"].mean()
    mean_att_safe = df[df["At_Risk"] == 0]["Attendance_Rate"].mean()
    insights.append(
        f"**Attendance Difference:** Students categorized as *Not At-Risk* have an average attendance rate of {mean_att_safe:.1f}%, compared to {mean_att_risk:.1f}% for *At-Risk* students."
    )
    
    # 4. Study Hours Distribution
    median_study_risk = df[df["At_Risk"] == 1]["Study_Hours"].median()
    median_study_safe = df[df["At_Risk"] == 0]["Study_Hours"].median()
    insights.append(
        f"**Study Hours Comparison:** Median study hours per week is {median_study_safe:.1f} hrs for *Not At-Risk* students versus {median_study_risk:.1f} hrs for *At-Risk* students."
    )
    
    return insights
