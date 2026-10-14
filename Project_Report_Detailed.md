# Statistical Machine Learning Mini Project Documentation
## Student Academic Performance & At-Risk Prediction Dashboard

---

### Project Metadata
- **Subject**: Statistical Machine Learning (SML) for Data Science
- **Project Title**: Predictive Modeling and Statistical Inference on Student Academic Performance & Early At-Risk Identification
- **Dataset**: Cohort of 1,200 students with 13 academic, behavioral, and demographic features
- **Core Technology Stack**: Python 3.12, Streamlit, Scikit-Learn, Statsmodels, SciPy, Pandas, NumPy, Matplotlib, Seaborn
- **Repository Location**: `c:\Users\aryan\Desktop\Aryan\Aryan Personal\Projects\SML`

---

## 1. Problem Statement

### 1.1 Academic Context & Dilemma
In modern educational institutions, student academic underperformance is frequently identified too late—typically after end-of-semester or summative examinations. At that stage, academic advisors and faculty have limited options for remediation, often leading to course failure, increased retake rates, academic probation, or student attrition. 

Traditional evaluation mechanisms rely on static checkpoints (e.g., final exams) rather than synthesized behavioral indicators. However, student performance is fundamentally a cumulative outcome influenced by early behavioral engagement (attendance rates, independent study hours, sleep habits), intermediate milestones (continuous internal assessments, prior semester grades), and socio-demographic factors (parental education levels, internet connectivity).

### 1.2 Core Problem Definition
Educational institutions lack an integrated, statistically sound, and transparent early-warning decision-support mechanism. Specifically, institutions face two distinct operational requirements:
1. **Continuous Score Estimation**: Accurately estimating a student's expected final performance (0–100 scale) to gauge overall cohort health and identify students performing below grade expectations.
2. **Binary Risk Detection Under Class Imbalance**: Proactively identifying students at risk of academic failure before summative examinations, where the at-risk segment is an operational minority (~18–20% of the cohort).

### 1.3 Operational Risk Rule
In this project, a student is formally operationalized as **"At-Risk" (`At_Risk = 1`)** if their projected or actual **Final Score falls strictly below 55.0 out of 100**. Students scoring $\ge 55.0$ are classified as **"On Track" (`At_Risk = 0`)**.

---

## 2. Project Objectives

### 2.1 Primary Goal
To formulate, validate, and deploy an end-to-end statistical and machine learning pipeline that provides both inferential understanding of academic performance drivers and actionable predictive capability for individual student risk detection.

### 2.2 Specific Objectives
1. **Exploratory Data Analysis & Bivariate Relationships**:
   - Assess univariate distributions, skewness, spread, and outlier boundaries across academic and behavioral variables.
   - Map correlation structures to isolate variables exhibiting strong linear associations with student scores.
2. **Inferential Statistics & Bootstrap Resampling**:
   - Apply non-parametric bootstrap resampling (2,000 iterations) to construct 95% empirical confidence intervals for population parameters (mean and median) without relying on strict parametric normality assumptions.
3. **Rigorous Hypothesis Testing**:
   - Formulate and test statistical hypotheses across key academic dimensions:
     - High Attendance ($\ge 80\%$) vs. Low Attendance ($< 80\%$).
     - High Study Hours ($\ge 15\text{ hrs/wk}$) vs. Low Study Hours ($< 15\text{ hrs/wk}$).
     - Extracurricular participation and remedial tutoring impact.
   - Verify statistical assumptions (Levene's test for homoscedasticity, Shapiro-Wilk for normality) and cross-validate findings using Welch’s t-test, Mann-Whitney U test, and non-parametric permutation tests.
4. **Predictive Modeling & Statistical Inference**:
   - **Ordinary Least Squares (OLS) Linear Regression**: Fit an interpretable linear model to predict continuous final scores, quantify feature weights ($\beta$), compute p-values and confidence intervals, and validate Gauss-Markov assumptions via residual diagnostics.
   - **Imbalance-Aware Classification (Logistic Regression / GLM Logit)**: Address minority-class imbalance using cost-sensitive learning (`class_weight='balanced'`), prioritizing Recall to minimize false negatives (failing to identify a struggling student).
5. **Interactive Decision-Support Deployment**:
   - Package all models, inference engines, and diagnostic charts into an intuitive, presentation-grade Streamlit web dashboard.

---

## 3. Dataset Description

### 3.1 Overview & Architecture
The study utilizes a structured cohort dataset consisting of **1,200 student records** across **13 features**. To reflect real-world educational characteristics accurately, the dataset models established multivariate empirical distributions (Beta for attendance, Gamma for study hours, Poisson for absences and tutoring, and Gaussian for previous academic standing).

### 3.2 Data Dictionary

| Variable Name | Data Type | Measurement Unit / Scale | Distribution / Properties | Description & Role |
| :--- | :--- | :--- | :--- | :--- |
| `Student_ID` | Categorical / String | Alphanumeric (STU1000–STU2199) | Identifier | Unique student tracking identifier (excluded from ML models). |
| `Study_Hours` | Numerical (Continuous) | Hours / week (3.0 – 35.0) | Gamma distributed ($\mu \approx 13.5$) | Self-reported weekly independent academic study time. |
| `Attendance_Rate` | Numerical (Continuous) | Percentage (40.0% – 100.0%) | Beta distributed ($\mu \approx 81.2\%$) | Overall lecture and lab attendance percentage. |
| `Previous_Grade` | Numerical (Continuous) | Percentage score (25.0 – 96.0) | Normal ($\mu \approx 59.5, \sigma \approx 15.0$) | Cumulative academic score from prior academic semester. |
| `Internal_Score` | Numerical (Continuous) | Score out of 100 (20.0 – 98.0) | Normal / Correlated ($\mu \approx 60.3$) | Mid-term tests, quizzes, and internal lab evaluations. |
| `Absences` | Numerical (Discrete) | Count of sessions (0 – 30) | Poisson distributed ($\lambda \propto [100 - \text{Att}]$) | Total missed classes over the term. |
| `Sleep_Hours` | Numerical (Continuous) | Hours / night (4.0 – 10.0) | Normal ($\mu \approx 6.9, \sigma \approx 1.2$) | Self-reported average nightly sleep duration. |
| `Tutoring_Sessions` | Numerical (Discrete) | Count / month (0 – 8) | Poisson distributed ($\lambda \approx 1.5$) | Remedial tutoring sessions attended. |
| `Parent_Education` | Categorical (Nominal) | 4 levels | High School (38%), Bachelor (40%), Master (17%), Doctorate (5%) | Highest educational attainment of parents/guardians. |
| `Extracurricular` | Categorical (Binary) | Yes / No | 46% Yes / 54% No | Active participation in clubs, sports, or societies. |
| `Internet_Access` | Categorical (Binary) | Yes / No | 85% Yes / 15% No | Reliable home broadband/internet connectivity. |
| `Final_Score` | Numerical (Continuous) | Score out of 100 (18.0 – 99.0) | Normal-like ($\mu = 65.41, \sigma = 11.42$) | **Regression Target**: Summative final examination score. |
| `At_Risk` | Categorical (Binary) | 0 (On Track), 1 (At-Risk) | Binary target ($\approx 18.4\%$ At-Risk) | **Classification Target**: Indicator for `Final_Score < 55.0`. |

### 3.3 Data Quality & Preprocessing Protocol
- **Missing Value & Duplicate Audit**: Zero missing values (`null = 0`) across all 13 columns; zero duplicate rows (`duplicates = 0`).
- **Feature Transformation**:
  - Continuous predictors are standardized using `StandardScaler` ($\mu = 0, \sigma = 1$) to enable direct coefficient comparison.
  - Categorical predictors (`Parent_Education`, `Extracurricular`, `Internet_Access`) are encoded via `OneHotEncoder(drop='first')` to prevent the dummy variable trap.
- **Strict Data Leakage Prevention**:
  - `Final_Score` is **strictly excluded** from the feature set $X$ during classification modeling. Predicting `At_Risk` from `Final_Score` directly would represent complete target leakage because the target was thresholded from `Final_Score`.

---

## 4. Exploratory Data Analysis (EDA)

### 4.1 Univariate Summary Statistics

| Feature | Count | Mean | Std Dev | Median (Q2) | IQR | Min | Max | Skewness | Kurtosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Study_Hours` | 1,200 | 13.52 | 6.78 | 12.30 | 8.90 | 3.00 | 35.00 | +0.642 | +0.118 |
| `Attendance_Rate` | 1,200 | 81.18 | 12.84 | 83.60 | 18.70 | 40.00 | 100.00 | -0.684 | -0.215 |
| `Previous_Grade` | 1,200 | 59.54 | 14.82 | 59.75 | 20.30 | 25.00 | 96.00 | -0.012 | -0.342 |
| `Internal_Score` | 1,200 | 60.31 | 14.15 | 60.10 | 19.50 | 20.00 | 98.00 | +0.035 | -0.288 |
| `Absences` | 1,200 | 5.24 | 4.12 | 4.00 | 5.00 | 0.00 | 24.00 | +1.185 | +1.240 |
| `Sleep_Hours` | 1,200 | 6.89 | 1.18 | 6.90 | 1.60 | 4.00 | 10.00 | +0.024 | -0.112 |
| `Tutoring_Sessions` | 1,200 | 1.52 | 1.25 | 1.00 | 2.00 | 0.00 | 7.00 | +0.945 | +0.682 |
| `Final_Score` | 1,200 | 65.41 | 11.42 | 65.45 | 15.60 | 24.80 | 97.40 | -0.154 | -0.210 |

### 4.2 Key Distributional Observations
1. **Target Normality**: The continuous target `Final_Score` follows an approximately symmetrical Gaussian distribution (mean = 65.41, median = 65.45, skewness = -0.154), making it well-suited for linear parametric modeling.
2. **Attendance Skewness**: Attendance exhibits moderate left-skewness (-0.684), with 75% of students maintaining attendance above 72.8%.
3. **Absences & Tutoring**: Both features exhibit right-skewed count profiles typical of Poisson arrival processes, where the majority have few absences/tutoring sessions, while a long tail represents highly disengaged or academically supported students.

### 4.3 Correlation Analysis (Pearson $r$ with `Final_Score`)

| Rank | Feature | Correlation ($r$) | Direction & Strength |
| :---: | :--- | :---: | :--- |
| 1 | `Previous_Grade` | **+0.72** | Very Strong Positive Linear Association |
| 2 | `Internal_Score` | **+0.68** | Strong Positive Linear Association |
| 3 | `Study_Hours` | **+0.54** | Moderate-to-Strong Positive Association |
| 4 | `Attendance_Rate` | **+0.48** | Moderate Positive Association |
| 5 | `Tutoring_Sessions` | **+0.12** | Weak Positive Association |
| 6 | `Sleep_Hours` | **+0.09** | Weak Positive Association |
| 7 | `Absences` | **-0.52** | Strong Negative Linear Association |

> **Methodological Note on Correlation vs. Causation**: While `Previous_Grade` and `Internal_Score` exhibit strong correlations with `Final_Score`, these metrics reflect sustained academic preparation and engagement rather than isolated causal switches.

### 4.4 Class Imbalance Analysis
- **Not At-Risk (`At_Risk = 0`)**: 979 students (**81.6%** of cohort).
- **At-Risk (`At_Risk = 1`)**: 221 students (**18.4%** of cohort).
- **Class Imbalance Ratio**: Approximately **4.4 : 1**.
- **Behavioral Profiling by Risk Group**:
  - *Not At-Risk Students*: Average Attendance = 84.1%, Median Study Hours = 14.5 hrs/wk, Average Absences = 4.2 days.
  - *At-Risk Students*: Average Attendance = 68.4%, Median Study Hours = 8.1 hrs/wk, Average Absences = 9.8 days.

---

## 5. Statistical Analysis (Sampling & Distribution)

### 5.1 Sampling Distribution & Central Limit Theorem Validation
To quantify the sampling uncertainty of the cohort mean without making unverifiable population normality assumptions, **non-parametric bootstrap resampling** was conducted with replacement over **$B = 2,000$ iterations** at a sample size of $n = 1,200$.

### 5.2 Bootstrap Estimation Results

```
Original Sample Mean (x̄): 65.41 points
Bootstrap Standard Error (SE_boot): 0.328 points
Bootstrap Mean Estimate: 65.41 points (Bias ≈ +0.002 points)
95% Empirical Percentile Confidence Interval: [64.77, 66.06] points

Original Sample Median: 65.45 points
95% Empirical Percentile Confidence Interval for Median: [64.80, 66.25] points
```

### 5.3 Inferential Takeaway
We are **95% confident** that the true unobserved population mean final score for this student body lies within the tight interval of **[64.77, 66.06]**. The standard error of only 0.328 points demonstrates high sample stability resulting from the $n = 1,200$ cohort size.

---

## 6. Hypothesis Testing

Four rigorous hypothesis tests were executed to evaluate whether observable student behaviors and interventions produce statistically significant differences in academic performance.

### 6.1 Test Summary Table

| Scenario | Group 1 ($G_1$) | Group 2 ($G_2$) | Mean Diff ($\Delta \bar{x}$) | Welch's $t$-stat | $p$-Value (Parametric) | Mann-Whitney $U$ $p$-Val | Permutation $p$-Val | Decision ($\alpha = 0.05$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Attendance** | Att $\ge 80\%$ ($n=608, \bar{x}=68.65$) | Att $< 80\%$ ($n=592, \bar{x}=62.08$) | **+6.57 pts** | $t = 10.39$ | $p = 2.87 \times 10^{-24}$ | $p = 2.72 \times 10^{-22}$ | $p = 0.0000$ | **Reject $H_0$** |
| **2. Study Hours** | Study $\ge 15$h ($n=444, \bar{x}=71.64$) | Study $< 15$h ($n=756, \bar{x}=61.75$) | **+9.89 pts** | $t = 15.94$ | $p = 5.15 \times 10^{-52}$ | $p = 4.01 \times 10^{-47}$ | $p = 0.0000$ | **Reject $H_0$** |
| **3. Extracurricular** | Participates ($n=532, \bar{x}=66.28$) | None ($n=668, \bar{x}=64.71$) | **+1.57 pts** | $t = 2.37$ | $p = 0.0180$ | $p = 0.0163$ | $p = 0.0165$ | **Reject $H_0$** |
| **4. Tutoring** | Tutoring $\ge 2$ ($n=587, \bar{x}=65.69$) | Tutoring $< 2$ ($n=613, \bar{x}=65.13$) | **+0.55 pts** | $t = 0.84$ | $p = 0.4008$ | $p = 0.3856$ | $p = 0.4235$ | **Fail to Reject $H_0$** |

---

### 6.2 Detailed Scenario Deep-Dive

#### Scenario 1: Attendance Rate Impact ($\ge 80\%$ vs. $< 80\%$)
- **Hypothesis Formulation**:
  - $H_0: \mu_{\text{high}} - \mu_{\text{low}} = 0$ (No difference in mean final score).
  - $H_1: \mu_{\text{high}} - \mu_{\text{low}} \neq 0$ (Statistically significant difference).
- **Assumption Verification**:
  - Levene’s Homogeneity of Variance Test: $F = 1.18, p = 0.278$ (Homoscedasticity assumption holds).
  - Shapiro-Wilk Normality Test: High group $p = 0.124$, Low group $p = 0.089$ (Normality plausible).
- **Statistical Decision**:
  - Welch’s $t = 10.389$, $p = 2.87 \times 10^{-24} < 0.05$.
  - **Conclusion**: Reject $H_0$. Students with $\ge 80\%$ attendance average **6.57 points higher** than students below 80%.

#### Scenario 2: Study Habits ($\ge 15\text{ hrs/wk}$ vs. $< 15\text{ hrs/wk}$)
- **Hypothesis Formulation**:
  - $H_0: \mu_{\ge 15\text{h}} = \mu_{< 15\text{h}}$ vs. $H_1: \mu_{\ge 15\text{h}} \neq \mu_{< 15\text{h}}$.
- **Statistical Decision**:
  - Welch’s $t = 15.938$, $p = 5.15 \times 10^{-52}$.
  - Observed difference = **+9.89 points**.
  - **Conclusion**: Overwhelming evidence to reject $H_0$. Weekly self-study of 15+ hours is one of the strongest empirical delineators of academic success.

#### Scenario 4: Tutoring Attendance ($\ge 2$ vs. $< 2$ Sessions/Month)
- **Observed Difference**: +0.55 points ($t = 0.841, p = 0.4008$).
- **Statistical Decision**: Fail to Reject $H_0$.
- **Methodological Insight**: In isolation, remedial tutoring attendance does not show statistical significance on final score. In educational contexts, this is a recognized phenomenon (selection bias / confounding): struggling students seek tutoring more frequently, which offsets the raw upward score comparison unless prior ability is controlled for via multivariable regression.

---

## 7. Regression and Classification / Discriminant Analysis

---

### 7.1 Part A: Ordinary Least Squares (OLS) Multiple Linear Regression

#### Model Formulation
$$Y_{\text{Final Score}} = \beta_0 + \sum_{j=1}^{p} \beta_j X_j + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

#### Goodness-of-Fit & Generalization Metrics (Train/Test: 80/20 Split)
- **Train $R^2$**: **0.8616** (86.16% of variance explained in training cohort)
- **Test $R^2$**: **0.8152** (81.52% of variance explained on unseen test cohort)
- **Adjusted $R^2$ (Test)**: **0.8054**
- **Mean Absolute Error (MAE)**: **3.88 points**
- **Root Mean Squared Error (RMSE)**: **4.73 points**

#### OLS Parameter Estimates & Statistical Significance (Statsmodels)

| Feature | Coefficient ($\beta$) | Std Error | $t$-Statistic | $p$-Value | 95% Confidence Interval |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Intercept** | **62.5918** | 0.415 | 150.84 | $< 0.0001$ | [61.78, 63.41] |
| `Previous_Grade` (std) | **+5.0613** | 0.208 | 24.28 | $< 0.0001$ | [4.65, 5.47] |
| `Internal_Score` (std) | **+3.8220** | 0.266 | 14.34 | $< 0.0001$ | [3.30, 4.35] |
| `Study_Hours` (std) | **+3.3497** | 0.192 | 17.45 | $< 0.0001$ | [2.97, 3.73] |
| `Attendance_Rate` (std) | **+1.6440** | 0.220 | 7.46 | $< 0.0001$ | [1.21, 2.08] |
| `Absences` (std) | **-1.3526** | 0.218 | -6.19 | $< 0.0001$ | [-1.78, -0.92] |
| `Tutoring_Sessions` (std) | **+0.9849** | 0.140 | 7.04 | $< 0.0001$ | [0.71, 1.26] |
| `Sleep_Hours` (std) | **+0.5939** | 0.139 | 4.26 | $< 0.0001$ | [0.32, 0.87] |
| `Parent_Edu: Doctorate` | **+3.2529** | 0.606 | 5.37 | $< 0.0001$ | [2.06, 4.44] |
| `Parent_Edu: Master` | **+2.2398** | 0.418 | 5.35 | $< 0.0001$ | [1.42, 3.06] |
| `Parent_Edu: High School` | **-2.6768** | 0.316 | -8.46 | $< 0.0001$ | [-3.30, -2.06] |
| `Internet_Access: Yes` | **+2.5866** | 0.402 | 6.44 | $< 0.0001$ | [1.80, 3.38] |
| `Extracurricular: Yes` | **+1.6407** | 0.281 | 5.84 | $< 0.0001$ | [1.09, 2.19] |

#### Regression Diagnostic Tests
1. **Normality of Residuals**: Shapiro-Wilk test on test residuals yielded $W = 0.993, p = 0.324$ ($p \ge 0.05$). Residuals conform to the normal error assumption.
2. **Homoscedasticity**: Breusch-Pagan Lagrange Multiplier test yielded $LM = 8.89, p = 0.712$ ($p \ge 0.05$). Variance of residuals is constant across fitted values.
3. **Multicollinearity**: Variance Inflation Factors (VIF) for all predictors remain below 2.8, well within the safe threshold of $< 5.0$.

---

### 7.2 Part B: Classification / Discriminant Analysis (Logistic Regression / GLM Logit)

#### Mathematical Formulation
$$\log\left(\frac{P(\text{At\_Risk}=1)}{1 - P(\text{At\_Risk}=1)}\right) = \beta_0 + \sum_{j=1}^{p} \beta_j X_j$$

#### The Imbalance Dilemma & Cost-Sensitive Resolution
Because only 18.4% of students are at risk, standard Maximum Likelihood Estimation creates an incentive to minimize loss on the majority class ("On Track"). In education, **missing an at-risk student (False Negative)** has catastrophic consequences compared to providing unnecessary tutoring to an on-track student (False Positive).

We compare a **Standard Baseline Classifier** against a **Cost-Sensitive Balanced Classifier** ($w_1 = \frac{N}{2 \cdot N_1} \approx 2.71$):

#### Classifier Comparison Table (Test Set: $n = 240$, At-Risk $n = 44$)

| Evaluation Metric | Baseline Model (`class_weight=None`) | Balanced Model (`class_weight='balanced'`) | Operational Significance |
| :--- | :---: | :---: | :--- |
| **Overall Accuracy** | 92.92% | 88.75% | Baseline looks deceptively higher due to majority class dominance. |
| **Recall (Sensitivity)** | **77.27%** | **95.45%** | **Balanced model captures 42 of 44 at-risk students (+18.2% boost).** |
| **False Negatives (Missed)** | **10 students missed** | **Only 2 students missed** | **80% reduction in dangerous false negative errors.** |
| **Precision** | 82.93% | 62.69% | Acceptable trade-off: slight increase in false alarms. |
| **Specificity** | 96.43% | 87.24% | Majority class recognition remains strong. |
| **F1-Score** | 0.8000 | 0.7568 | Harmonic mean reflects recall trade-off. |
| **ROC-AUC Score** | **0.9730** | **0.9745** | Superb class-discriminative ranking power across both models. |

#### Confusion Matrix Comparison

```
Baseline Confusion Matrix:
               Predicted On-Track    Predicted At-Risk
Actual On-Track       189                    7
Actual At-Risk         10 [DANGEROUS]       34

Balanced Confusion Matrix:
               Predicted On-Track    Predicted At-Risk
Actual On-Track       171                   25
Actual At-Risk          2 [MINIMIZED]       42
```

#### Logistic Regression Odds Ratios ($\text{OR} = \exp(\beta)$)

| Feature | Log-Odds ($\beta$) | Odds Ratio ($\exp(\beta)$) | Plain-Language Risk Interpretation |
| :--- | :---: | :---: | :--- |
| `Parent_Edu: High School` | +0.919 | **2.507** | **+150.7% higher odds** of being At-Risk vs. baseline education. |
| `Absences` | +0.559 | **1.749** | **+74.9% higher odds** of At-Risk per 1-SD increase in absences. |
| `Sleep_Hours` | -0.201 | **0.818** | **-18.2% lower odds** of At-Risk per 1-SD increase in regular sleep. |
| `Tutoring_Sessions` | -0.476 | **0.621** | **-37.9% lower odds** of At-Risk when controlling for other covariates. |
| `Extracurricular: Yes` | -0.496 | **0.609** | **-39.1% lower odds** of At-Risk for extracurricular participants. |
| `Attendance_Rate` | -0.681 | **0.506** | **-49.4% lower odds** of At-Risk per 1-SD increase in attendance. |
| `Internet_Access: Yes` | -0.724 | **0.485** | **-51.5% lower odds** of At-Risk with home internet. |
| `Study_Hours` | -1.103 | **0.332** | **-66.8% lower odds** of At-Risk per 1-SD increase in study time. |
| `Internal_Score` | -1.778 | **0.169** | **-83.1% lower odds** of At-Risk per 1-SD increase in internal marks. |
| `Previous_Grade` | -1.820 | **0.162** | **-83.8% lower odds** of At-Risk per 1-SD increase in prior semester grade. |

---

## 8. Results and Interpretation

### 8.1 Key Academic Findings
1. **Academic Momentum Dominates**: `Previous_Grade` ($\beta = 5.06$) and `Internal_Score` ($\beta = 3.82$) are the strongest linear determinants of final score. This confirms that academic performance is cumulative rather than sudden.
2. **Actionable Behavioral Levers**:
   - Every 1-SD increase in weekly study hours increases final expected score by **+3.35 points** and cuts at-risk odds by **66.8%**.
   - Absences impose a direct linear penalty of **-1.35 points** per standard deviation and increase at-risk odds by **+74.9%**.
3. **The Tutoring Paradox Resolved**: While univariate hypothesis testing showed no significant difference between students taking $\ge 2$ tutoring sessions vs. $< 2$ ($p = 0.40$), multivariable regression revealed that holding prior ability and attendance constant, tutoring contributes a positive **+0.98 points** ($p < 0.0001$) and reduces risk odds by **37.9%**. This demonstrates the classic necessity of controlling for confounding variables.
4. **Recall Priority in Educational Decision Support**: Choosing the Balanced Logistic Regression model increases detection from 77.3% to **95.5%**, successfully protecting 8 additional vulnerable students who would have slipped through undetected under a standard model.

### 8.2 Threats to Validity & Limitations
- **Synthetic Cohort Structure**: The dataset is statistically simulated to mimic real universities; empirical deployment would require calibration on real student records.
- **Observational Nature**: Regression coefficients reflect conditional expectations rather than causal interventions. Recommending more study hours must be accompanied by actual study skills support.

---

## 9. Conclusion & Recommendations

### 9.1 Summary of Project Outcomes
- **Statistical Pipeline**: Completed univariate EDA, bootstrap resampling with 95% CIs, and 4 formal hypothesis tests with non-parametric and permutation validations.
- **Predictive Performance**:
  - Continuous OLS Regression achieved $R^2 = 0.815$ and $\text{MAE} = 3.88$ points with fully satisfied Gauss-Markov assumptions.
  - Imbalance-Aware Classification achieved **95.5% Recall** and **ROC-AUC = 0.975** for at-risk detection.
- **Decision Support Tool**: Streamlit interactive application provides immediate risk classification, key-driver breakdowns, and individual student simulations.

### 9.2 Practical Institutional Recommendations
1. **Early Trigger at 4 Weeks**: Flag any student whose attendance drops below 80% or who accrues $\ge 3$ absences in the first month.
2. **Mandatory Midterm Review**: Students scoring below 55 in internal assessments should automatically receive targeted academic coaching before final examinations.
3. **Advisory Simulation Use**: Use the student predictor tab in student mentoring sessions to demonstrate the tangible score improvement that an additional 4 hours of weekly study would yield.

---

## Appendix A: Python & Jupyter Notebook Structure

```text
Student-Performance-Dashboard/
│
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Project dependencies
├── PROJECT_DOCUMENTATION.md    # Comprehensive academic report
│
├── data/
│   ├── generate_data.py        # Reproducible synthetic dataset generator
│   └── student_data.csv        # Cohort records (1,200 rows × 13 columns)
│
├── src/
│   ├── __init__.py             # Module initialization
│   ├── data_preprocessing.py   # Audits, standard scaling, and one-hot encoding
│   ├── eda.py                  # Descriptive stats and correlation computation
│   ├── bootstrap_analysis.py   # Resampling (2000 runs) and 95% Confidence Intervals
│   ├── hypothesis_testing.py   # Welch's t-test, Mann-Whitney U, Permutation tests
│   ├── regression_model.py     # OLS regression with Statsmodels diagnostics
│   ├── classification_model.py # Baseline vs Balanced Logistic Regression models
│   └── utils.py                # Styling and UI helper components
│
└── models/
    ├── regression_model.pkl    # Serialized regression pipeline artifact
    └── classification_model.pkl# Serialized classification pipeline artifact
```

---

## Appendix B: 5–7 Minute Oral Presentation Blueprint

| Time | Slide / Topic | Content & Talking Points | Visual / Chart |
| :---: | :--- | :--- | :--- |
| **0:00 - 1:00** | **Slide 1: Problem & Motivation** | - Academic failure is identified too late.<br>- Need early warning for at-risk students ($< 55$ points).<br>- Dataset: 1,200 students, 13 features. | Streamlit KPI header (1,200 students, 18.4% at-risk). |
| **1:00 - 2:00** | **Slide 2: EDA & Statistical Insights** | - Mean score: 65.4 pts, 18.4% at-risk imbalance.<br>- Strongest positive correlations: Prior grades ($r = +0.72$), internal score ($r = +0.68$), study hours ($r = +0.54$).<br>- Strongest negative: Absences ($r = -0.52$). | Distribution histogram & Correlation bar chart. |
| **2:00 - 3:15** | **Slide 3: Sampling & Hypothesis Testing** | - Bootstrap 95% CI: $[64.77, 66.06]$ pts ($\text{SE} = 0.328$).<br>- Attendance Test ($\ge 80\%$ vs $< 80\%$): Difference = $+6.57$ pts ($p = 2.87 \times 10^{-24}$).<br>- Study Hours Test ($\ge 15$h vs $< 15$h): Difference = $+9.89$ pts ($p = 5.15 \times 10^{-52}$).<br>- Permutation & Mann-Whitney U confirm parametric tests. | Boxplots comparing Attendance and Study Hour segments. |
| **3:15 - 4:45** | **Slide 4: Dual Modeling (OLS & Classification)** | - **OLS Regression**: Test $R^2 = 0.815$, $\text{MAE} = 3.88$ pts. Residuals normal ($p=0.32$) & homoscedastic ($p=0.71$).<br>- **Imbalanced Classification**: Baseline misses 10 students (77.3% recall). Balanced model catches 95.5% (only 2 missed!). ROC-AUC = 0.975. | Confusion matrix comparison & ROC curve. |
| **4:45 - 6:00** | **Slide 5: Live Dashboard Demonstration** | - Show interactive Streamlit dashboard.<br>- Demo single-student predictor: adjust study hours and show real-time risk drop and score gain. | Live screen-share of Streamlit web interface. |
| **6:00 - 7:00** | **Slide 6: Conclusion & Q&A** | - Prioritize Recall in educational risk models.<br>- Early 4-week interventions for low attendance.<br>- Open floor for faculty questions. | Summary recommendations slide. |
