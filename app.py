"""A simple, presentation-ready student performance dashboard."""

import sys

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Streamlit renders charts in the browser, so Matplotlib must not use a desktop GUI backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__" and not st.runtime.exists():
    import streamlit.web.cli as stcli

    sys.argv = ["streamlit", "run", sys.argv[0]]
    sys.exit(stcli.main())

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.classification_model import train_classification_models
from src.data_preprocessing import load_data
from src.regression_model import train_linear_regression
from src.utils import apply_custom_styles, render_metric_card

apply_custom_styles()

plt.style.use("default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10,
    "axes.labelcolor": "#29465B",
    "xtick.color": "#5B6B7B",
    "ytick.color": "#5B6B7B",
    "axes.edgecolor": "#C8D9E2",
    "grid.color": "#E2ECF1",
    "grid.alpha": 0.8,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
})


@st.cache_data
def get_dataset():
    return load_data()


@st.cache_data
def get_models(data):
    return train_linear_regression(data), train_classification_models(data)


def transform_for_prediction(prep, input_data):
    """Keep transformed feature names so scikit-learn inference remains warning-free."""
    transformed = prep["preprocessor"].transform(input_data)
    return pd.DataFrame(
        transformed,
        columns=prep["transformed_features"],
        index=input_data.index,
    )


df = get_dataset()
reg_results, clf_results = get_models(df)
mean_score = df["Final_Score"].mean()
at_risk_count = int(df["At_Risk"].sum())
at_risk_pct = at_risk_count / len(df) * 100

st.sidebar.markdown("""
<div style="padding: .2rem 0 1rem; border-bottom: 1px solid #C8D9E2; margin-bottom: 1rem;">
  <h2 style="margin:0; color:#102A43; font-family:'Playfair Display', Georgia, serif;">Student Analytics</h2>
  <p style="margin:.35rem 0 0; color:#3D6A78; font-size:.76rem; letter-spacing:.08em; text-transform:uppercase; font-weight:600;">Simple dashboard</p>
</div>
<div style="padding:.85rem; background:#FFFFFF; border:1px solid #CFE3E1; border-radius:8px; color:#29465B; font-size:.84rem; line-height:1.65;">
  <b>Cohort:</b> 1,200 students<br>
  <b>Risk rule:</b> Final score below 55<br>
  <b style="color:#0F766E;">● Models ready</b>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Student Performance Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">A clear view of cohort performance, student risk, and key drivers.</div>', unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    render_metric_card("Students", f"{len(df):,}", "Current cohort")
with kpi2:
    render_metric_card("Average score", f"{mean_score:.1f} / 100", "Across all students")
with kpi3:
    render_metric_card("Students at risk", f"{at_risk_pct:.1f}%", f"{at_risk_count:,} students", delta_color="#D94F4F")
with kpi4:
    render_metric_card("Risk detection", f"{clf_results['metrics_balanced']['recall'] * 100:.0f}%", "At-risk students found", delta_color="#0F766E")

tab_overview, tab_drivers, tab_predictor, tab_data = st.tabs([
    "Overview", "What affects performance?", "Student predictor", "Data summary"
])

with tab_overview:
    st.markdown("### Cohort performance at a glance")
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df["Final_Score"], bins=24, kde=True, color="#0F766E", alpha=.38, edgecolor="#FFFFFF", ax=ax)
        ax.axvline(55, color="#D94F4F", linestyle="--", linewidth=2, label="At-risk score (55)")
        ax.axvline(mean_score, color="#17324D", linewidth=2, label=f"Average ({mean_score:.1f})")
        ax.set_title("Final score distribution")
        ax.set_xlabel("Final score")
        ax.set_ylabel("Number of students")
        ax.legend(frameon=False)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        safe_count = len(df) - at_risk_count
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(["On track", "At risk"], [safe_count, at_risk_count], color=["#0F766E", "#D94F4F"], width=.55)
        for bar, value in zip(bars, [safe_count, at_risk_count]):
            ax.text(bar.get_x() + bar.get_width()/2, value + 18, f"{value:,}", ha="center", fontweight="bold", color="#17324D")
        ax.set_title("Students needing attention")
        ax.set_ylabel("Number of students")
        ax.set_ylim(0, max(safe_count, at_risk_count) * 1.16)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### Key takeaways")
    st.markdown(f"""
    <div class="alert-safe"><b>Most students are on track:</b> {100 - at_risk_pct:.1f}% are expected to score 55 or above.</div>
    <div class="alert-risk"><b>Focus area:</b> {at_risk_count:,} students fall below the 55-point risk threshold and may benefit from early support.</div>
    <div class="alert-moderate"><b>How to use this dashboard:</b> use the next tab to explain the biggest drivers, then test an individual student in the predictor.</div>
    """, unsafe_allow_html=True)

with tab_drivers:
    st.markdown("### The factors most connected to final score")
    numeric_columns = ["Study_Hours", "Attendance_Rate", "Previous_Grade", "Internal_Score", "Absences", "Sleep_Hours", "Tutoring_Sessions"]
    correlations = df[numeric_columns + ["Final_Score"]].corr(numeric_only=True)["Final_Score"].drop("Final_Score").sort_values()
    labels = correlations.index.str.replace("_", " ")
    colors = ["#D94F4F" if value < 0 else "#0F766E" for value in correlations.values]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(labels, correlations.values, color=colors)
    ax.axvline(0, color="#17324D", linewidth=1)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Relationship with final score (correlation)")
    ax.set_title("Positive means scores tend to rise as the factor increases")
    for index, value in enumerate(correlations.values):
        ax.text(value + (0.025 if value >= 0 else -0.025), index, f"{value:.2f}", va="center", ha="left" if value >= 0 else "right", color="#29465B")
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Correlation shows association, not proof that one factor causes another.")

    top_positive = correlations.idxmax().replace("_", " ")
    top_negative = correlations.idxmin().replace("_", " ")
    st.markdown(f"""
    <div class="edu-card"><div class="edu-title">Simple explanation</div>
    <div class="edu-body"><b>{top_positive}</b> has the strongest positive relationship with final score. <b>{top_negative}</b> has the strongest negative relationship. These are useful starting points for student support conversations.</div></div>
    """, unsafe_allow_html=True)

with tab_predictor:
    st.markdown("### Estimate one student's result")
    st.caption("Enter the information available for a student. The model gives an estimated final score and an early-risk signal.")
    with st.form("student_prediction_form"):
        main_inputs, support_inputs = st.columns(2)
        with main_inputs:
            in_study = st.slider("Study hours per week", 3.0, 35.0, 14.0, .5)
            in_att = st.slider("Attendance rate (%)", 40.0, 100.0, 75.0, 1.0)
            in_prev = st.slider("Previous grade", 25.0, 98.0, 62.0, 1.0)
            in_intern = st.slider("Internal assessment score", 20.0, 98.0, 65.0, 1.0)
        with support_inputs:
            in_abs = st.number_input("Absences (days)", 0, 30, 4, 1)
            in_sleep = st.slider("Sleep per night (hours)", 4.0, 10.0, 7.0, .5)
            in_tutor = st.number_input("Tutoring sessions per month", 0, 8, 1, 1)
            in_parent = st.selectbox("Parent education", ["High School", "Bachelor", "Master", "Doctorate"], index=1)
            in_extra = st.selectbox("Extracurricular activity", ["Yes", "No"])
            in_net = st.selectbox("Internet access", ["Yes", "No"])
        submitted = st.form_submit_button("Estimate student outcome", use_container_width=True)

    input_df = pd.DataFrame([{
        "Study_Hours": in_study, "Attendance_Rate": in_att, "Previous_Grade": in_prev,
        "Internal_Score": in_intern, "Absences": in_abs, "Sleep_Hours": in_sleep,
        "Tutoring_Sessions": in_tutor, "Parent_Education": in_parent,
        "Extracurricular": in_extra, "Internet_Access": in_net,
    }])
    reg_input = transform_for_prediction(reg_results["prep"], input_df)
    clf_input = transform_for_prediction(clf_results["prep"], input_df)
    pred_score = float(reg_results["model"].predict(reg_input)[0])
    risk_probability = float(clf_results["clf_balanced"].predict_proba(clf_input)[0, 1])
    pred_score = np.clip(pred_score, 0, 100)
    risk_label = "High risk" if risk_probability > .60 else "Moderate risk" if risk_probability >= .30 else "Low risk"
    alert_class = "alert-risk" if risk_probability > .60 else "alert-moderate" if risk_probability >= .30 else "alert-safe"
    recommendation = "Arrange an academic check-in soon." if risk_probability > .60 else "Monitor progress and consider additional support." if risk_probability >= .30 else "Maintain the current learning plan."

    if submitted:
        score_col, risk_col, action_col = st.columns(3)
        with score_col:
            render_metric_card("Estimated final score", f"{pred_score:.1f} / 100", "Model estimate")
        with risk_col:
            render_metric_card("Risk probability", f"{risk_probability * 100:.0f}%", "Chance of scoring below 55", delta_color="#D94F4F" if risk_probability >= .30 else "#0F766E")
        with action_col:
            render_metric_card("Risk level", risk_label, "Early warning signal")
        st.markdown(f'<div class="{alert_class}"><b>Suggested next step:</b> {recommendation}</div>', unsafe_allow_html=True)

        risk_factors = []
        protective_factors = []
        if in_study <= 5:
            risk_factors.append(f"low study time ({in_study:.0f} hours/week)")
        elif in_study >= 15:
            protective_factors.append(f"strong study time ({in_study:.0f} hours/week)")
        if in_att < 70:
            risk_factors.append(f"low attendance ({in_att:.0f}%)")
        elif in_att >= 75:
            protective_factors.append(f"steady attendance ({in_att:.0f}%)")
        if in_prev < 55:
            risk_factors.append(f"a low previous grade ({in_prev:.0f})")
        elif in_prev >= 60:
            protective_factors.append(f"a solid previous grade ({in_prev:.0f})")
        if in_intern < 55:
            risk_factors.append(f"a low internal score ({in_intern:.0f})")
        elif in_intern >= 60:
            protective_factors.append(f"a strong internal score ({in_intern:.0f})")
        if in_abs >= 8:
            risk_factors.append(f"frequent absences ({in_abs} days)")
        elif in_abs <= 2:
            protective_factors.append(f"very few absences ({in_abs} days)")

        risk_text = ", ".join(risk_factors) if risk_factors else "no major warning signs in the selected inputs"
        protective_text = ", ".join(protective_factors) if protective_factors else "few strong protective signals"
        st.markdown(f"""
        <div class="edu-card">
          <div class="edu-title">Why this result?</div>
          <div class="edu-body"><b>Factors that raise concern:</b> {risk_text}.<br>
          <b>Factors that lower risk:</b> {protective_text}.<br><br>
          The model weighs the complete profile. A low study-hours value is important, but it does not by itself determine the risk level.</div>
        </div>
        """, unsafe_allow_html=True)

with tab_data:
    st.markdown("### Essential cohort data")
    summary = df[["Final_Score", "Study_Hours", "Attendance_Rate", "Previous_Grade", "Internal_Score", "Absences"]].describe().T.reset_index()
    summary = summary.rename(columns={"index": "Measure", "count": "Students", "mean": "Average", "std": "Variation", "min": "Lowest", "max": "Highest"})
    st.dataframe(summary[["Measure", "Average", "Variation", "Lowest", "Highest"]], use_container_width=True, hide_index=True)
    st.info("For privacy, individual student records and dataset downloads are not available in the live dashboard. Only cohort-level summaries are shown.")
