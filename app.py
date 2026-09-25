"""
Streamlit UI for the Student Marks Prediction project.
Run:  streamlit run app.py
Needs the files created by the notebook in the model/ folder.
"""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = "model"

st.set_page_config(page_title="Student Marks Predictor", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    artifacts = joblib.load(os.path.join(MODEL_DIR, "artifacts.pkl"))
    return model, scaler, artifacts


def grade_for(marks):
    if marks >= 90:
        return "A+"
    if marks >= 80:
        return "A"
    if marks >= 70:
        return "B"
    if marks >= 60:
        return "C"
    if marks >= 50:
        return "D"
    if marks >= 40:
        return "E (Pass)"
    return "F (Fail)"


def prepare_input(raw, artifacts):
    """Apply the same encoding, column order and scaling used in training."""
    df = pd.DataFrame([raw])
    df = pd.get_dummies(df, columns=list(artifacts["categorical_features"].keys()))
    df = df.reindex(columns=artifacts["feature_columns"], fill_value=0)
    return df.astype(float)


LABELS = {
    "study_hours": "Study hours per day",
    "attendance_pct": "Attendance (%)",
    "previous_exam_marks": "Previous exam marks (out of 100)",
    "assignment_marks": "Assignment marks (out of 100)",
    "internal_marks": "Internal marks (out of 100)",
    "practice_test_score": "Practice test score (out of 100)",
    "sleep_hours": "Sleep hours per day",
    "commute_minutes": "Commute time (minutes)",
    "gender": "Gender",
    "parental_education": "Parental education",
    "internet_access": "Internet access at home",
    "extracurricular": "Extracurricular activities",
}

st.title("Student Final Marks Predictor")
st.write("Enter the student's details to predict the final exam marks "
         "(Linear Regression model).")

if not os.path.exists(os.path.join(MODEL_DIR, "model.pkl")):
    st.error("model/model.pkl not found. Run the notebook first to train and save the model.")
    st.stop()

model, scaler, artifacts = load_artifacts()

with st.form("student_form"):
    raw = {}
    col1, col2 = st.columns(2)
    items = list(artifacts["numeric_features"].items())
    for i, (col, cfg) in enumerate(items):
        target_col = col1 if i % 2 == 0 else col2
        raw[col] = target_col.slider(
            LABELS.get(col, col),
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            value=float(cfg["default"]),
            step=float(cfg["step"]),
        )
    for col, options in artifacts["categorical_features"].items():
        raw[col] = st.selectbox(LABELS.get(col, col), options)
    submitted = st.form_submit_button("Predict final marks")

if submitted:
    X = prepare_input(raw, artifacts)
    X_scaled = scaler.transform(X)
    pred = float(np.clip(model.predict(X_scaled)[0], 0, 100))

    st.subheader("Prediction")
    c1, c2 = st.columns(2)
    c1.metric("Predicted final marks", f"{pred:.1f} / 100")
    c2.metric("Expected grade", grade_for(pred))

    rmse = artifacts.get("metrics", {}).get("test_rmse")
    if rmse is not None:
        st.caption(f"Typical error of the model on unseen data: about ±{rmse:.1f} marks (test RMSE).")

    st.subheader("Input summary")
    summary = pd.DataFrame(
        {"Feature": [LABELS.get(k, k) for k in raw], "Value": [str(v) for v in raw.values()]}
    )
    st.table(summary)

with st.expander("About the model"):
    m = artifacts.get("metrics", {})
    if m:
        st.write(f"Test R²: **{m['test_r2']:.3f}**, MAE: {m['test_mae']:.2f}, "
                 f"RMSE: {m['test_rmse']:.2f}")
    st.write("Features used:", ", ".join(artifacts["feature_columns"]))
