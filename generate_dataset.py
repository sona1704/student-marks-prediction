"""
Generate a synthetic student-performance dataset for the
Student Marks Prediction project (BCS OJT - Project 2).

The data is realistic but artificial. It deliberately contains:
  - missing values in a few columns
  - duplicate rows
  - a handful of outliers / data-entry errors
  - one feature (commute_minutes) and one categorical (gender) with no real
    effect on marks, so feature selection has something to remove

Run:  python generate_dataset.py      -> writes student_marks.csv
"""
import numpy as np
import pandas as pd


def generate_student_data(n=1000, seed=42):
    rng = np.random.default_rng(seed)

    # Hidden "ability" drives the correlated academic scores
    ability = rng.normal(0, 1, n)

    gender = rng.choice(["Male", "Female"], n)
    parental_education = rng.choice(
        ["High School", "Bachelor", "Master", "PhD"], n, p=[0.35, 0.40, 0.18, 0.07]
    )
    internet_access = rng.choice(["Yes", "No"], n, p=[0.82, 0.18])
    extracurricular = rng.choice(["Yes", "No"], n, p=[0.45, 0.55])

    study_hours = np.clip(rng.normal(5 + 0.8 * ability, 2.0), 0.5, 12)
    attendance_pct = np.clip(rng.normal(80 + 5 * ability, 9), 40, 100)
    previous_exam_marks = np.clip(rng.normal(65 + 10 * ability, 7), 20, 100)
    assignment_marks = np.clip(rng.normal(70 + 6 * ability, 10), 20, 100)
    internal_marks = np.clip(rng.normal(64 + 8 * ability, 8), 15, 100)
    practice_test_score = np.clip(rng.normal(62 + 9 * ability, 9), 10, 100)
    sleep_hours = np.clip(rng.normal(7, 1.2, n), 3, 10)
    commute_minutes = np.clip(rng.normal(30, 15, n), 2, 90)  # irrelevant feature

    edu_effect = pd.Series(parental_education).map(
        {"High School": 0.0, "Bachelor": 1.0, "Master": 2.0, "PhD": 3.0}
    ).to_numpy()

    final_exam_marks = (
        -8
        + 0.25 * previous_exam_marks
        + 0.15 * internal_marks
        + 0.08 * assignment_marks
        + 0.15 * practice_test_score
        + 2.0 * study_hours
        + 0.15 * attendance_pct
        + 1.0 * sleep_hours
        + edu_effect
        + np.where(internet_access == "Yes", 2.0, 0.0)
        + np.where(extracurricular == "Yes", 1.0, 0.0)
        + rng.normal(0, 4, n)
    )
    final_exam_marks = np.clip(final_exam_marks, 0, 100)

    df = pd.DataFrame({
        "student_id": [f"S{i:04d}" for i in range(1, n + 1)],
        "gender": gender,
        "parental_education": parental_education,
        "internet_access": internet_access,
        "extracurricular": extracurricular,
        "study_hours": study_hours.round(1),
        "attendance_pct": attendance_pct.round(1),
        "previous_exam_marks": previous_exam_marks.round(0),
        "assignment_marks": assignment_marks.round(0),
        "internal_marks": internal_marks.round(0),
        "practice_test_score": practice_test_score.round(0),
        "sleep_hours": sleep_hours.round(1),
        "commute_minutes": commute_minutes.round(0),
        "final_exam_marks": final_exam_marks.round(1),
    })

    # Outliers / data-entry errors
    idx = rng.choice(n, 11, replace=False)
    df.loc[idx[:5], "study_hours"] = rng.uniform(25, 40, 5).round(1)
    df.loc[idx[5:8], "attendance_pct"] = rng.uniform(120, 150, 3).round(1)
    df.loc[idx[8:], "sleep_hours"] = rng.uniform(18, 20, 3).round(1)

    # Missing values (~3% in selected columns)
    for col in ["study_hours", "attendance_pct", "sleep_hours",
                "assignment_marks", "parental_education", "internet_access"]:
        miss = rng.choice(n, int(0.03 * n), replace=False)
        df.loc[miss, col] = np.nan

    # Duplicate rows
    dups = df.sample(20, random_state=seed)
    df = pd.concat([df, dups], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    data = generate_student_data()
    data.to_csv("student_marks.csv", index=False)
    print("Saved student_marks.csv with shape", data.shape)
