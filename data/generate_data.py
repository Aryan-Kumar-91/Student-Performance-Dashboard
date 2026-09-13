"""
Dataset Generator for Student Academic Performance & At-Risk Prediction
Subject: Statistical Machine Learning for Data Science

Generates a realistic, statistically coherent synthetic dataset of 1,200 student records
with realistic multivariate distributions, subtle Gaussian noise, and known ground-truth
relationships suitable for regression, hypothesis testing, bootstrap resampling, and classification.
"""

import os
import numpy as np
import pandas as pd

def generate_student_dataset(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic student academic performance dataset.
    
    Parameters:
        n_samples (int): Number of student observations (default: 1200).
        random_state (int): Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: Processed student performance dataframe.
    """
    np.random.seed(random_state)
    
    # 1. Base Identifiers
    student_ids = [f"STU{1000 + i}" for i in range(n_samples)]
    
    # 2. Continuous & Discrete Predictor Variables
    # Study hours per week (Gamma distribution, 4 to 35 hours)
    study_hours = np.clip(np.random.gamma(shape=3.0, scale=4.5, size=n_samples), 3.0, 35.0).round(1)
    
    # Attendance Rate (Beta distribution skewed towards high attendance, 40% - 100%)
    raw_attendance = np.random.beta(a=4.5, b=2.2, size=n_samples)
    attendance_rate = np.clip(40.0 + raw_attendance * 60.0, 40.0, 100.0).round(1)
    
    # Absences (Poisson distribution inversely correlated with attendance)
    absences = np.clip(np.random.poisson(lam=(100.0 - attendance_rate) / 3.8), 0, 30)
    
    # Previous Semester Grade (Normal distribution centered at 60, sd=15)
    previous_grade = np.clip(np.random.normal(loc=59.5, scale=15.0, size=n_samples), 25.0, 96.0).round(1)
    
    # Internal Assessment Score (Correlated with study hours and previous grade)
    raw_internal = (
        0.50 * previous_grade + 
        0.90 * study_hours + 
        0.25 * attendance_rate -
        0.40 * absences +
        np.random.normal(loc=0, scale=7.0, size=n_samples)
    )
    internal_assessment = np.clip(raw_internal * (92.0 / 115.0), 20.0, 98.0).round(1)
    
    # Sleep Hours per Day (Normal distribution centered at 6.8, sd=1.2)
    sleep_hours = np.clip(np.random.normal(loc=6.9, scale=1.2, size=n_samples), 4.0, 10.0).round(1)
    
    # Tutoring Sessions per Month (Poisson distribution, lam=1.5)
    tutoring_sessions = np.clip(np.random.poisson(lam=1.5, size=n_samples), 0, 8)
    
    # Categorical Predictors
    parent_edu_choices = ["High School", "Bachelor", "Master", "Doctorate"]
    parent_edu_probs = [0.38, 0.40, 0.17, 0.05]
    parent_education = np.random.choice(parent_edu_choices, size=n_samples, p=parent_edu_probs)
    
    extracurricular = np.random.choice(["Yes", "No"], size=n_samples, p=[0.46, 0.54])
    internet_access = np.random.choice(["Yes", "No"], size=n_samples, p=[0.85, 0.15])
    
    # 3. Continuous Target Generation: Final Academic Score (0 - 100 scale)
    # Ground-truth statistical linear-additive relationship with realistic noise
    parent_edu_bonus = {
        "High School": 0.0,
        "Bachelor": 2.8,
        "Master": 4.5,
        "Doctorate": 6.2
    }
    edu_effect = np.array([parent_edu_bonus[pe] for pe in parent_education])
    extra_effect = np.where(extracurricular == "Yes", 2.0, 0.0)
    internet_effect = np.where(internet_access == "Yes", 2.5, 0.0)
    
    raw_final_score = (
        5.5 +                                       # Intercept
        0.35 * previous_grade +                     # Strong predictor
        0.32 * internal_assessment +                # Strong predictor
        0.48 * study_hours +                        # Positive impact
        0.16 * attendance_rate -                    # Positive impact
        0.42 * absences +                           # Penalty
        0.75 * tutoring_sessions +                  # Small positive
        0.45 * (sleep_hours - 7.0) +                # Optimal sleep benefit
        edu_effect + extra_effect + internet_effect +
        np.random.normal(loc=0, scale=4.5, size=n_samples) # Residual error term
    )
    
    final_score = np.clip(raw_final_score, 18.0, 99.0).round(1)
    
    # 4. Binary Target Generation: At-Risk Classification
    # Academic Threshold: Final Score < 55.0 is categorized as At-Risk (1), else Not At-Risk (0)
    # This creates a realistic academic at-risk proportion (~20% - 25% of cohort)
    at_risk = (final_score < 55.0).astype(int)
    
    # Assemble DataFrame
    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Study_Hours": study_hours,
        "Attendance_Rate": attendance_rate,
        "Previous_Grade": previous_grade,
        "Internal_Score": internal_assessment,
        "Absences": absences,
        "Sleep_Hours": sleep_hours,
        "Tutoring_Sessions": tutoring_sessions,
        "Parent_Education": parent_education,
        "Extracurricular": extracurricular,
        "Internet_Access": internet_access,
        "Final_Score": final_score,
        "At_Risk": at_risk
    })
    
    return df

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "student_data.csv")
    
    df = generate_student_dataset(n_samples=1200, random_state=42)
    df.to_csv(output_path, index=False)
    
    print(f"[SUCCESS] Generated {len(df)} student records at '{output_path}'.")
    print(f"Feature Columns: {list(df.columns)}")
    print(f"At-Risk Class Counts:\n{df['At_Risk'].value_counts(normalize=True).round(3) * 100}%")
    print(f"Final Score Stats:\n{df['Final_Score'].describe().round(2)}")
