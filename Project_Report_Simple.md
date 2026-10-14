# Student Performance Analytics & Risk Prediction
## Mini Project Documentation (Tables & Practical Results Edition)

---

### Project Metadata
- **Project Title**: Predictive Modeling and Statistical Inference on Student Academic Performance & Early At-Risk Identification
- **Subject**: Statistical Machine Learning (SML) for Data Science
- **Cohort Size**: 1,200 Students | 13 Attributes | 0 Missing Values
- **Core Models**: OLS Multiple Linear Regression & Cost-Sensitive Balanced Logistic Regression
- **Web App**: Interactive Streamlit Dashboard (`app.py`)

---

## 1. Problem Statement

- **The Problem**: In academic institutions, struggling students are typically identified after semester exams, when remediation is no longer possible.
- **The Solution**: An early-warning data system using continuous behavioral signals (attendance, study hours, absences) and early assessments (internal marks, prior grades) to predict outcomes and flag vulnerable students well in advance.
- **Operational Risk Rule**: A student is classified as **At-Risk (`At_Risk = 1`)** if their Final Score is strictly **below 55 out of 100**. Students scoring $\ge 55$ are classified as **On Track (`At_Risk = 0`)**.

---

## 2. Project Objectives

1. **Exploratory Data Analysis**: Profile student distributions, correlations, and performance disparities.
2. **Statistical Uncertainty (Bootstrap)**: Construct 95% Confidence Intervals for population parameters without parametric assumptions.
3. **Hypothesis Testing**: Statistically test whether attendance ($\ge 80\%$) and study hours ($\ge 15\text{ hrs}$) significantly boost academic marks.
4. **Continuous Score Prediction**: Fit an Ordinary Least Squares (OLS) Regression model to forecast final scores (0–100 scale).
5. **Imbalance-Aware Classification**: Train a cost-sensitive classifier prioritizing **Recall** to minimize missed at-risk students.
6. **Decision-Support Tool**: Deploy an interactive Streamlit dashboard for real-time faculty advisory and student simulation.

---

## 3. Dataset Description

The dataset comprises **1,200 student records** across **13 features** with zero missing values.

### Table 1: Dataset Attributes & Data Dictionary

| Feature Name | Data Type | Range / Scale | Distribution / Properties | Description & Role |
| :--- | :--- | :---: | :--- | :--- |
| `Student_ID` | Identifier | STU1000 – STU2199 | Discrete IDs | Unique student tracking identifier (excluded from models). |
| `Study_Hours` | Continuous | 3.0 – 35.0 hrs/wk | Gamma ($\mu = 13.52$) | Self-reported weekly independent academic study time. |
| `Attendance_Rate` | Continuous | 40.0% – 100.0% | Beta ($\mu = 81.18\%$) | Overall class and lab attendance percentage. |
| `Previous_Grade` | Continuous | 25.0 – 96.0 marks | Normal ($\mu = 59.54$) | Cumulative score from the preceding academic semester. |
| `Internal_Score` | Continuous | 20.0 – 98.0 marks | Normal ($\mu = 60.31$) | Midterm tests, quizzes, and internal lab evaluations. |
| `Absences` | Discrete | 0 – 24 classes | Poisson ($\mu = 5.24$) | Missed lectures throughout the semester. |
| `Sleep_Hours` | Continuous | 4.0 – 10.0 hrs/day | Normal ($\mu = 6.89$) | Average daily sleep duration. |
| `Tutoring_Sessions` | Discrete | 0 – 7 sessions/mo | Poisson ($\mu = 1.52$) | Remedial tutoring sessions attended. |
| `Parent_Education` | Categorical | 4 Levels | High School (38%), Bachelor (40%), Master (17%), PhD (5%) | Highest educational attainment of parents/guardians. |
| `Extracurricular` | Binary | Yes / No | 46% Yes / 54% No | Participation in university clubs, sports, or societies. |
| `Internet_Access` | Binary | Yes / No | 85% Yes / 15% No | Reliable high-speed home internet connectivity. |
| `Final_Score` | Continuous | 18.0 – 99.0 marks | Normal-like ($\mu = 65.41$) | **Regression Target**: Summative final examination score. |
| `At_Risk` | Binary | 0 or 1 | 81.6% Safe / 18.4% Risk | **Classification Target**: Indicator for `Final_Score < 55.0`. |

> **Data Leakage Safeguard**: `Final_Score` is strictly excluded from predictor inputs $X$ during classification to prevent target leakage.

---

## 4. Exploratory Data Analysis (EDA)

### Table 2: Cohort Profile & Correlation with Final Score

| Feature | Whole Cohort ($n=1,200$) | On Track ($\ge 55, n=979$) | At-Risk ($< 55, n=221$) | Correlation with Final Score ($r$) | Direction & Strength |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Final_Score`** | **65.41 ± 11.42** | **68.85 ± 9.32** | **49.88 ± 4.21** | **1.00** | Target Variable |
| **`Previous_Grade`** | 59.54 ± 14.82 | 62.81 ± 13.51 | 44.92 ± 11.20 | **+0.72** | Very Strong Positive Association |
| **`Internal_Score`** | 60.31 ± 14.15 | 63.50 ± 12.80 | 45.92 ± 10.60 | **+0.68** | Strong Positive Association |
| **`Study_Hours`** | 13.52 ± 6.78 | 14.73 ± 6.55 | 8.16 ± 4.62 | **+0.54** | Moderate-Strong Positive Association |
| **`Attendance_Rate`** | 81.18% ± 12.84% | 84.08% ± 10.82% | 68.35% ± 13.50% | **+0.48** | Moderate Positive Association |
| **`Absences`** | 5.24 ± 4.12 | 4.21 ± 3.40 | 9.80 ± 4.50 | **-0.52** | Strong Negative Linear Association |
| **`Tutoring_Sessions`** | 1.52 ± 1.25 | 1.54 ± 1.26 | 1.45 ± 1.20 | **+0.12** | Weak Positive Association |
| **`Sleep_Hours`** | 6.89 ± 1.18 | 6.94 ± 1.15 | 6.67 ± 1.25 | **+0.09** | Weak Positive Association |

### Key EDA Takeaways
- **Class Imbalance**: 221 At-Risk students (**18.4%**) vs. 979 On-Track students (**81.6%**) $\approx$ 4.4 : 1 imbalance.
- **Vulnerability Gap**: At-risk students study **45% fewer hours** (8.2 vs 14.7 hrs/wk) and accumulate **more than double the absences** (9.8 vs 4.2 days).

---

## 5. Statistical Analysis & Confidence Intervals

To estimate population parameters without assuming normality, **non-parametric bootstrap resampling** was performed with $B = 2,000$ iterations.

### Table 3: Bootstrap Resampling Parameter Estimates ($B = 2,000$)

| Parameter | Sample Statistic | Bootstrap Std Error ($SE$) | 95% Confidence Interval | Method |
| :--- | :---: | :---: | :---: | :--- |
| **Mean Final Score** | **65.41 marks** | **0.328 marks** | **[64.77, 66.06] marks** | Percentile Bootstrap |
| **Median Final Score** | **65.45 marks** | **0.365 marks** | **[64.80, 66.25] marks** | Percentile Bootstrap |

**Interpretation**: We are **95% confident** that the true average final score for this student population lies between **64.77 and 66.06 marks**.

---

## 6. Hypothesis Testing

Four hypothesis tests were evaluated at significance level $\alpha = 0.05$.

### Table 4: Formal Hypothesis Testing Results

| Test Scenario | Group 1 ($G_1$) | Group 2 ($G_2$) | Score Difference | Welch's $t$-stat | Parametric $p$-Val | Mann-Whitney $p$-Val | Permutation $p$-Val | Decision ($\alpha=0.05$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Attendance** | Att $\ge 80\%$ ($n=608, \bar{x}=68.65$) | Att $< 80\%$ ($n=592, \bar{x}=62.08$) | **+6.57 marks** | $t = 10.39$ | $p = 2.87 \times 10^{-24}$ | $p = 2.72 \times 10^{-22}$ | $p = 0.0000$ | **Reject $H_0$** |
| **2. Study Hours** | Study $\ge 15$h ($n=444, \bar{x}=71.64$) | Study $< 15$h ($n=756, \bar{x}=61.75$) | **+9.89 marks** | $t = 15.94$ | $p = 5.15 \times 10^{-52}$ | $p = 4.01 \times 10^{-47}$ | $p = 0.0000$ | **Reject $H_0$** |
| **3. Extracurricular** | Participates ($n=532, \bar{x}=66.28$) | None ($n=668, \bar{x}=64.71$) | **+1.57 marks** | $t = 2.37$ | $p = 0.0180$ | $p = 0.0163$ | $p = 0.0165$ | **Reject $H_0$** |
| **4. Tutoring** | Tutoring $\ge 2$ ($n=587, \bar{x}=65.69$) | Tutoring $< 2$ ($n=613, \bar{x}=65.13$) | **+0.55 marks** | $t = 0.84$ | $p = 0.4008$ | $p = 0.3856$ | $p = 0.4235$ | **Fail to Reject $H_0$** |

### Insights:
- **Attendance & Study Hours**: Massive, unquestionable statistical significance ($p < 10^{-23}$).
- **The Tutoring Paradox**: In simple univariate testing, tutoring appears insignificant ($p = 0.40$) because struggling students take more tutoring (confounding bias). Multivariable regression separates this effect.

---

## 7. Predictive Modeling & Model Results

---

### Part A: Predicting Final Marks (OLS Linear Regression)

### Table 5: Regression Model Performance (Train/Test: 80/20)

| Metric | Training Cohort ($n=960$) | Testing Cohort ($n=240$) | Interpretation |
| :--- | :---: | :---: | :--- |
| **$R^2$ (Variance Explained)** | **86.16%** | **81.52%** | High explanatory power with minimal overfitting |
| **Adjusted $R^2$** | **85.98%** | **80.54%** | Adjusted for number of feature parameters |
| **Mean Absolute Error (MAE)** | **3.51 marks** | **3.88 marks** | Average prediction error is under 4 marks |
| **Root Mean Squared Error (RMSE)** | **4.41 marks** | **4.73 marks** | Penalizes larger individual errors |
| **Residual Normality Test (Shapiro-Wilk)** | - | **$p = 0.324$** | **Passes**: Residuals are normally distributed ($p \ge 0.05$) |
| **Homoscedasticity (Breusch-Pagan)** | - | **$p = 0.712$** | **Passes**: Residual variance is constant ($p \ge 0.05$) |

### Table 6: OLS Feature Coefficients & Statistical Significance

| Feature (Standardized) | Coefficient ($\beta$) | Std Error | $t$-Statistic | $p$-Value | 95% Confidence Interval | Effect on Final Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Intercept** | **62.59** | 0.415 | 150.84 | $< 0.0001$ | [61.78, 63.41] | Baseline average score |
| `Previous_Grade` | **+5.06** | 0.208 | 24.28 | $< 0.0001$ | [4.65, 5.47] | **+5.1 marks** per 1-SD increase |
| `Internal_Score` | **+3.82** | 0.266 | 14.34 | $< 0.0001$ | [3.30, 4.35] | **+3.8 marks** per 1-SD increase |
| `Study_Hours` | **+3.35** | 0.192 | 17.45 | $< 0.0001$ | [2.97, 3.73] | **+3.4 marks** per 1-SD increase |
| `Parent_Edu: Doctorate` | **+3.25** | 0.606 | 5.37 | $< 0.0001$ | [2.06, 4.44] | **+3.3 marks** vs baseline |
| `Internet_Access: Yes` | **+2.59** | 0.402 | 6.44 | $< 0.0001$ | [1.80, 3.38] | **+2.6 marks** vs no internet |
| `Parent_Edu: Master` | **+2.24** | 0.418 | 5.35 | $< 0.0001$ | [1.42, 3.06] | **+2.2 marks** vs baseline |
| `Attendance_Rate` | **+1.64** | 0.220 | 7.46 | $< 0.0001$ | [1.21, 2.08] | **+1.6 marks** per 1-SD increase |
| `Extracurricular: Yes` | **+1.64** | 0.281 | 5.84 | $< 0.0001$ | [1.09, 2.19] | **+1.6 marks** vs no activities |
| `Tutoring_Sessions` | **+0.98** | 0.140 | 7.04 | $< 0.0001$ | [0.71, 1.26] | **+1.0 marks** per 1-SD increase |
| `Sleep_Hours` | **+0.59** | 0.139 | 4.26 | $< 0.0001$ | [0.32, 0.87] | **+0.6 marks** per 1-SD increase |
| `Absences` | **-1.35** | 0.218 | -6.19 | $< 0.0001$ | [-1.78, -0.92] | **-1.4 marks penalty** per 1-SD increase |
| `Parent_Edu: High School` | **-2.68** | 0.316 | -8.46 | $< 0.0001$ | [-3.30, -2.06] | **-2.7 marks penalty** vs degree |

---

### Part B: Detecting At-Risk Students (Logistic Regression / GLM)

### Table 7: Classification Model Comparison (Holdout Test Set: $n = 240$)

| Evaluation Metric | Baseline Model (`class_weight=None`) | Balanced Model (`class_weight='balanced'`) | Practical Significance in Education |
| :--- | :---: | :---: | :--- |
| **Overall Accuracy** | 92.92% | 88.75% | Baseline looks high due to majority "Safe" class |
| **Recall (At-Risk Detection)** | **77.27%** | **95.45%** | **Balanced model catches 42 of 44 at-risk students!** |
| **False Negatives (Missed Students)**| **10 students missed** | **Only 2 students missed** | **80% reduction in dangerous missed failures** |
| **Precision** | 82.93% | 62.69% | Acceptable trade-off: slight rise in false alarms |
| **Specificity** | 96.43% | 87.24% | High correct identification of on-track students |
| **F1-Score** | 0.8000 | 0.7568 | Harmonic mean reflecting recall balance |
| **ROC-AUC Score** | **0.9730** | **0.9745** | Superb discriminatory capability across all cutoffs |

### Table 8: Confusion Matrix Breakdown (Test Set: 196 Safe, 44 At-Risk)

| Model Scenario | True Negative (Correct On-Track) | False Positive (False Alarm) | False Negative (Missed Failure) | True Positive (Detected At-Risk) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Classifier** | 189 | 7 | **10 [DANGEROUS]** | 34 |
| **Balanced Classifier (Recommended)** | 171 | 25 | **2 [MINIMIZED]** | **42 [CAPTURED]** |

### Table 9: Odds Ratios ($\text{OR} = \exp(\beta)$) & Risk Multipliers

| Predictor Variable | Log-Odds ($\beta$) | Odds Ratio ($\exp(\beta)$) | Impact on Odds of Being At-Risk |
| :--- | :---: | :---: | :--- |
| `Parent_Edu: High School` | +0.919 | **2.507** | **+150.7% higher odds** of being At-Risk vs baseline degree |
| `Absences` | +0.559 | **1.749** | **+74.9% higher odds** per 1-SD increase in missed classes |
| `Sleep_Hours` | -0.201 | **0.818** | **-18.2% lower odds** with healthy regular sleep |
| `Tutoring_Sessions` | -0.476 | **0.621** | **-37.9% lower odds** with tutoring when controlling for ability |
| `Extracurricular: Yes` | -0.496 | **0.609** | **-39.1% lower odds** for extracurricular participants |
| `Attendance_Rate` | -0.681 | **0.506** | **-49.4% lower odds** per 1-SD increase in attendance |
| `Internet_Access: Yes` | -0.724 | **0.485** | **-51.5% lower odds** with home broadband connectivity |
| `Study_Hours` | -1.103 | **0.332** | **-66.8% lower odds** per 1-SD increase in study hours |
| `Internal_Score` | -1.778 | **0.169** | **-83.1% lower odds** per 1-SD increase in midterm marks |
| `Previous_Grade` | -1.820 | **0.162** | **-83.8% lower odds** per 1-SD increase in past grade |

---

## 8. Results & Interpretation

1. **Academic Momentum**: `Previous_Grade` ($\beta = 5.06$) and `Internal_Score` ($\beta = 3.82$) explain the largest share of variance. Student failure is rarely sudden; warning signals exist early in the semester.
2. **Behavioral Levers**: While faculty cannot change past grades, **study hours** ($+3.35$ marks per SD) and **absences** ($-1.35$ marks penalty) are directly modifiable through advising.
3. **Recall Over Precision**: In educational early warning, a **False Negative** (failing to support an at-risk student) can result in course failure or dropout. A **False Positive** (offering tutoring to a borderline student) causes zero harm. The **Balanced Classifier with 95.5% Recall** is the optimal institutional choice.

---

## 9. Conclusion & Recommendations

### Actionable Institutional Protocol
1. **Week 4 Early Trigger**: Automatically flag any student whose attendance drops below 80% or who accrues 3+ absences in the first month.
2. **Mandatory Midterm Review**: Provide structured peer-tutoring for any student scoring below 55 in midterm internal assessments.
3. **Advisory Simulation**: Use the Streamlit dashboard in mentoring sessions to demonstrate how 4 additional weekly study hours reduces failure probability by over 60%.

---

## Quick Presentation Cheat-Sheet (5 Minutes)

- **Min 0–1**: **Goal** – Predict final scores and detect at-risk students ($< 55$ marks) before final examinations.
- **Min 1–2**: **Data & EDA** – 1,200 students, 18.4% at-risk. Strongest drivers: Past grade ($r=0.72$), Internal ($r=0.68$), Study hours ($r=0.54$), Absences ($r=-0.52$).
- **Min 2–3**: **Hypothesis Testing** – Statistically proved: Attendance $\ge 80\%$ adds $+6.6$ marks ($p < 0.001$); Study hours $\ge 15$ adds $+9.9$ marks ($p < 0.001$).
- **Min 3–4**: **Predictive Models** – OLS Regression achieves $R^2 = 81.5\%$ ($\text{MAE} = 3.88$ marks). Balanced Logistic Regression achieves **95.5% Recall**, reducing missed failures from 10 down to 2.
- **Min 4–5**: **Live Demo & Protocol** – Streamlit dashboard demonstration and the 4-week early attendance trigger protocol.
