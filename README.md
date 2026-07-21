# 🚢 Titanic Survival Predictor

Predicts whether a Titanic passenger would have survived, based on class, age, sex, fare, and family size — using a Random Forest classifier with SHAP explainability. Built as Project 1 of my 75-day DS/AI roadmap.

🌐 **Live App:** [Click to open](https://vijendrapokharkar15-design-titanic-survival-predicto-app-wbdy2y.streamlit.app)

---

## Problem

The Titanic dataset is the classic intro to binary classification, but the goal here wasn't just a notebook with a score — it was a tool a non-technical person could open, play with, and understand *why* it predicted what it did. That's why this became a full 4-tab Streamlit app with SHAP explanations built in, not just a model.

## Approach

- Engineered features: title extracted from name (Mr/Mrs/Miss/Master), family size, cabin presence, log-transformed fare.
- Trained and compared three models: Random Forest, Logistic Regression, and Gradient Boosting.
- Random Forest was selected for deployment based on CV performance.
- Used SHAP to explain individual predictions rather than just reporting aggregate feature importance.

## Features

- 🔮 **Predict** — Enter passenger details, get survival probability + SHAP explanation instantly
- 📊 **Data Insights** — Visual exploration of Titanic dataset patterns
- 🤖 **Model Comparison** — Random Forest vs Logistic Regression vs Gradient Boosting
- 📁 **Batch Predict** — Upload a CSV, download predictions for all passengers

---

## Model Performance

| Metric | Value |
|--------|-------|
| CV AUC | 0.875 ± 0.016 |
| Test AUC | 0.849 |
| Test Accuracy | 81.6% |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-0.51-green)

---

## Key Findings

- **Gender** is the strongest predictor — females had 74.2% survival vs 18.9% for males
- **Class** matters — 1st class: 63%, 2nd: 47.3%, 3rd: 24.2%
- **Family size 4** had the highest survival (72.4%) — solo travellers only 30.4%
- **SHAP** confirmed `Title_Mr` and `Sex_encoded` as the top two features

---

## Project Structure

- `app.py` — Streamlit web application (4 tabs)
- `train_model.py` — ML training pipeline with feature engineering
- `requirements.txt` — Pinned dependencies
- `titanic.csv` — Training dataset (891 passengers)
- `model/rf_model.pkl` — Trained Random Forest
- `model/features.pkl` — Feature list

---

## How to Run Locally

1. Clone the repo — `git clone https://github.com/vijendrapokharkar15-design/titanic-survival-predictor.git`
2. Install dependencies — `pip install -r requirements.txt`
3. Train the model — `python train_model.py`
4. Launch the app — `streamlit run app.py`
5. Open browser at `http://localhost:8501`

---

## Part of DS-AI-75D Journey

This project is part of my [75-Day Data Science & AI Roadmap](https://github.com/vijendrapokharkar15-design/DS-AI-75D) — built on Days 30–32 using techniques from Phase 2 (Days 16–29).
