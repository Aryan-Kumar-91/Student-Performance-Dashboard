"""
Module: Bootstrap Resampling & Confidence Intervals
Subject: Statistical Machine Learning for Data Science

Implements non-parametric bootstrap resampling with replacement to estimate sampling distributions,
standard errors, and empirical confidence intervals without strict parametric assumptions.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

def run_bootstrap_mean(
    data: np.ndarray, 
    n_iterations: int = 2000, 
    confidence_level: float = 0.95, 
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Performs bootstrap resampling to estimate the sampling distribution and CI of the mean.
    
    Parameters:
        data: 1D array of observed sample values.
        n_iterations: Number of bootstrap resamples (default: 2000).
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).
        random_state: Seed for reproducibility.
        
    Returns:
        Dict containing original statistic, bootstrap distribution, standard error, CI bounds, and interpretation.
    """
    rng = np.random.default_rng(random_state)
    n = len(data)
    original_stat = float(np.mean(data))
    
    # Resample with replacement n_iterations times
    boot_samples = rng.choice(data, size=(n_iterations, n), replace=True)
    boot_means = np.mean(boot_samples, axis=1)
    
    # Calculate Percentile Confidence Interval
    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0
    
    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))
    boot_se = float(np.std(boot_means, ddof=1))
    boot_bias = float(np.mean(boot_means) - original_stat)
    
    interpretation = (
        f"The sample mean Final Score is {original_stat:.2f}. Based on {n_iterations:,} bootstrap resamples, "
        f"we are {int(confidence_level*100)}% confident that the true population mean Final Academic Score "
        f"lies within the interval [{ci_lower:.2f}, {ci_upper:.2f}], with an estimated standard error of {boot_se:.3f}."
    )
    
    return {
        "statistic_name": "Mean Final Score",
        "original_statistic": original_stat,
        "bootstrap_distribution": boot_means,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
        "bootstrap_se": boot_se,
        "bootstrap_bias": boot_bias,
        "n_iterations": n_iterations,
        "interpretation": interpretation
    }

def run_bootstrap_median(
    data: np.ndarray, 
    n_iterations: int = 2000, 
    confidence_level: float = 0.95, 
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Performs bootstrap resampling to estimate the sampling distribution and CI of the median.
    """
    rng = np.random.default_rng(random_state)
    n = len(data)
    original_stat = float(np.median(data))
    
    boot_samples = rng.choice(data, size=(n_iterations, n), replace=True)
    boot_medians = np.median(boot_samples, axis=1)
    
    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0
    
    ci_lower = float(np.percentile(boot_medians, lower_pct))
    ci_upper = float(np.percentile(boot_medians, upper_pct))
    boot_se = float(np.std(boot_medians, ddof=1))
    boot_bias = float(np.mean(boot_medians) - original_stat)
    
    interpretation = (
        f"The sample median Final Score is {original_stat:.2f}. Based on {n_iterations:,} bootstrap resamples, "
        f"we are {int(confidence_level*100)}% confident that the true population median Final Academic Score "
        f"lies within the interval [{ci_lower:.2f}, {ci_upper:.2f}]."
    )
    
    return {
        "statistic_name": "Median Final Score",
        "original_statistic": original_stat,
        "bootstrap_distribution": boot_medians,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
        "bootstrap_se": boot_se,
        "bootstrap_bias": boot_bias,
        "n_iterations": n_iterations,
        "interpretation": interpretation
    }

def run_bootstrap_diff_means(
    group1: np.ndarray, 
    group2: np.ndarray, 
    group1_name: str = "Group 1", 
    group2_name: str = "Group 2", 
    n_iterations: int = 2000, 
    confidence_level: float = 0.95, 
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Performs two-sample bootstrap resampling for the difference in means (Group 1 - Group 2).
    """
    rng = np.random.default_rng(random_state)
    n1, n2 = len(group1), len(group2)
    
    mean1 = float(np.mean(group1))
    mean2 = float(np.mean(group2))
    original_diff = mean1 - mean2
    
    boot1 = rng.choice(group1, size=(n_iterations, n1), replace=True)
    boot2 = rng.choice(group2, size=(n_iterations, n2), replace=True)
    
    boot_diffs = np.mean(boot1, axis=1) - np.mean(boot2, axis=1)
    
    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_diffs, (alpha / 2.0) * 100.0))
    ci_upper = float(np.percentile(boot_diffs, (1.0 - alpha / 2.0) * 100.0))
    boot_se = float(np.std(boot_diffs, ddof=1))
    
    interpretation = (
        f"The observed difference in mean score ({group1_name} - {group2_name}) is {original_diff:.2f} points. "
        f"With {int(confidence_level*100)}% confidence, the true difference in population means lies between "
        f"[{ci_lower:.2f}, {ci_upper:.2f}]. "
        f"{'Since 0 is not included in the interval, there is a statistically significant difference between the two groups.' if (ci_lower > 0 or ci_upper < 0) else 'Since 0 is contained within the interval, we cannot conclude a significant difference at this confidence level.'}"
    )
    
    return {
        "statistic_name": f"Mean Difference ({group1_name} vs {group2_name})",
        "original_statistic": original_diff,
        "mean1": mean1,
        "mean2": mean2,
        "group1_name": group1_name,
        "group2_name": group2_name,
        "bootstrap_distribution": boot_diffs,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
        "bootstrap_se": boot_se,
        "n_iterations": n_iterations,
        "interpretation": interpretation
    }
