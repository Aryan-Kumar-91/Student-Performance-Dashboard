"""
Module: Hypothesis Testing
Subject: Statistical Machine Learning for Data Science

Performs statistical hypothesis testing with assumption checking (normality & homoscedasticity),
parametric tests (Welch's t-test), non-parametric alternatives (Mann-Whitney U), and permutation tests.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats

def check_normality(data: np.ndarray, group_name: str = "Sample") -> Dict[str, Any]:
    """
    Checks normality assumption using Shapiro-Wilk (or D'Agostino-Pearson for n > 5000).
    """
    # Use Shapiro-Wilk if n <= 5000 else normaltest
    if len(data) <= 5000:
        stat, p_val = stats.shapiro(data)
        test_name = "Shapiro-Wilk Test"
    else:
        stat, p_val = stats.normaltest(data)
        test_name = "D'Agostino's K² Test"
        
    is_normal = bool(p_val >= 0.05)
    return {
        "group": group_name,
        "test_name": test_name,
        "statistic": float(stat),
        "p_value": float(p_val),
        "is_normal": is_normal,
        "interpretation": f"Data for {group_name} {'follows' if is_normal else 'deviates from'} a normal distribution (p = {p_val:.4f})."
    }

def check_homoscedasticity(group1: np.ndarray, group2: np.ndarray) -> Dict[str, Any]:
    """
    Checks equality of variances using Levene's Test (robust to non-normality).
    """
    stat, p_val = stats.levene(group1, group2)
    equal_var = bool(p_val >= 0.05)
    return {
        "test_name": "Levene's Test for Homogeneity of Variance",
        "statistic": float(stat),
        "p_value": float(p_val),
        "equal_variance": equal_var,
        "interpretation": f"Variances are {'equal (homoscedastic)' if equal_var else 'significantly different (heteroscedastic)'} across groups (p = {p_val:.4f})."
    }

def run_hypothesis_test(
    df: pd.DataFrame,
    test_scenario: str = "attendance",
    alpha: float = 0.05,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Runs a structured hypothesis test based on meaningful academic student segments.
    
    Supported scenarios:
    1. 'attendance': High Attendance (>=80%) vs Low Attendance (<80%)
    2. 'study_hours': High Study Hours (>=15 hrs/wk) vs Low Study Hours (<15 hrs/wk)
    3. 'extracurricular': Extracurricular Activities (Yes vs No)
    4. 'tutoring': Tutoring Support (>=2 sessions) vs Low Tutoring (<2 sessions)
    """
    if test_scenario == "attendance":
        g1_mask = df["Attendance_Rate"] >= 80.0
        g1_label = "High Attendance (≥ 80%)"
        g2_label = "Low Attendance (< 80%)"
        variable_tested = "Attendance Rate"
    elif test_scenario == "study_hours":
        g1_mask = df["Study_Hours"] >= 15.0
        g1_label = "High Study Hours (≥ 15 hrs/wk)"
        g2_label = "Low Study Hours (< 15 hrs/wk)"
        variable_tested = "Study Hours"
    elif test_scenario == "extracurricular":
        g1_mask = df["Extracurricular"] == "Yes"
        g1_label = "Participates in Extracurriculars"
        g2_label = "No Extracurriculars"
        variable_tested = "Extracurricular Participation"
    elif test_scenario == "tutoring":
        g1_mask = df["Tutoring_Sessions"] >= 2
        g1_label = "Regular Tutoring (≥ 2 sessions/mo)"
        g2_label = "Low/No Tutoring (< 2 sessions/mo)"
        variable_tested = "Tutoring Sessions"
    else:
        raise ValueError(f"Unknown test scenario: {test_scenario}")
        
    g1_scores = df.loc[g1_mask, "Final_Score"].dropna().values
    g2_scores = df.loc[~g1_mask, "Final_Score"].dropna().values
    
    n1, n2 = len(g1_scores), len(g2_scores)
    m1, m2 = float(np.mean(g1_scores)), float(np.mean(g2_scores))
    s1, s2 = float(np.std(g1_scores, ddof=1)), float(np.std(g2_scores, ddof=1))
    
    # 1. Assumption Audits
    norm1 = check_normality(g1_scores, g1_label)
    norm2 = check_normality(g2_scores, g2_label)
    levene_res = check_homoscedasticity(g1_scores, g2_scores)
    
    # 2. Parametric Test: Welch's Two-Sample t-test (does not assume equal variance)
    ttest_res = stats.ttest_ind(g1_scores, g2_scores, equal_var=levene_res["equal_variance"])
    t_stat = float(ttest_res.statistic)
    t_pval = float(ttest_res.pvalue)
    
    # 3. Non-Parametric Test: Mann-Whitney U Test (Rank-Sum)
    mwu_res = stats.mannwhitneyu(g1_scores, g2_scores, alternative="two-sided")
    mwu_stat = float(mwu_res.statistic)
    mwu_pval = float(mwu_res.pvalue)
    
    # 4. Permutation Test for Difference in Means (Resampling under Null)
    rng = np.random.default_rng(random_state)
    observed_diff = m1 - m2
    pooled_data = np.concatenate([g1_scores, g2_scores])
    n_perm = 2000
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(pooled_data)
        perm_diffs[i] = np.mean(shuffled[:n1]) - np.mean(shuffled[n1:])
    perm_pval = float(np.mean(np.abs(perm_diffs) >= np.abs(observed_diff)))
    
    # Decision Logic
    reject_null = t_pval < alpha
    
    h0_statement = f"There is NO significant difference in population mean Final Score between {g1_label} and {g2_label} (μ₁ = μ₂)."
    h1_statement = f"There IS a statistically significant difference in population mean Final Score between {g1_label} and {g2_label} (μ₁ ≠ μ₂)."
    
    if reject_null:
        decision_text = f"Reject Null Hypothesis (H₀) at α = {alpha} (p = {t_pval:.4e} < {alpha})."
        interpretation = (
            f"The data provides sufficient statistical evidence that students in '{g1_label}' (Mean = {m1:.2f}) "
            f"achieve a significantly different average Final Score compared to students in '{g2_label}' (Mean = {m2:.2f}), "
            f"t = {t_stat:.2f}, p < {alpha}. "
            f"Important note: While this difference is statistically significant, this represents an empirical association and does not independently establish direct causality."
        )
    else:
        decision_text = f"Fail to Reject Null Hypothesis (H₀) at α = {alpha} (p = {t_pval:.4f} ≥ {alpha})."
        interpretation = (
            f"There is insufficient statistical evidence at the α = {alpha} significance level to reject H₀. "
            f"The observed difference in sample means ({observed_diff:.2f} points) may be attributable to random sampling variability."
        )
        
    return {
        "scenario": test_scenario,
        "variable_tested": variable_tested,
        "g1_label": g1_label,
        "g2_label": g2_label,
        "n1": n1,
        "n2": n2,
        "m1": m1,
        "m2": m2,
        "s1": s1,
        "s2": s2,
        "observed_diff": observed_diff,
        "h0": h0_statement,
        "h1": h1_statement,
        "alpha": alpha,
        "normality_g1": norm1,
        "normality_g2": norm2,
        "homoscedasticity": levene_res,
        "t_stat": t_stat,
        "t_pval": t_pval,
        "mwu_stat": mwu_stat,
        "mwu_pval": mwu_pval,
        "perm_pval": perm_pval,
        "reject_null": reject_null,
        "decision": decision_text,
        "interpretation": interpretation,
        "g1_scores": g1_scores,
        "g2_scores": g2_scores,
        "perm_diffs": perm_diffs
    }
