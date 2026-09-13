# Student Performance Dashboard

A simple Streamlit dashboard for understanding student performance and identifying students who may need academic support.

## What it shows

- Cohort size, average score, at-risk percentage, and risk-detection recall
- Final-score distribution and the at-risk threshold
- Factors most associated with final score
- A single-student predictor with a plain-language explanation of the result
- Aggregate data summaries only

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

## Project structure

```text
app.py                  # Streamlit dashboard
src/                    # Data preparation and ML models
data/generate_data.py   # Reproducible synthetic dataset generator
requirements.txt        # Python dependencies
```

## Data and privacy

The dashboard uses a synthetic dataset generated on first run from `data/generate_data.py`. The live dashboard intentionally exposes only cohort-level summaries; it does not offer raw student-record downloads.

## Security

- CORS and XSRF protection are enabled in the Streamlit configuration.
- Model artifacts are generated locally and excluded from version control.
- Do not add real student data, secrets, or API keys to the repository.

## Notes

Predictions are decision-support estimates, not final academic decisions. The risk model evaluates the full student profile rather than any individual input in isolation.
