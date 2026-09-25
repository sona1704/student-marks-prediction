# Student Marks Prediction (BCS OJT – Project 2)

Predicts a student's **final exam marks** (0–100) from study hours, attendance, previous exam marks, assignment marks, internal marks, practice test score, sleep hours and a few background columns, using **Linear Regression**.

## Links

- **Live app (Streamlit):** https://students-mark-prediction.streamlit.app/
- **GitHub repo:** https://github.com/sona1704/student-marks-prediction
- **Open the notebook in Colab:** https://colab.research.google.com/github/sona1704/student-marks-prediction/blob/main/Student_Marks_Prediction.ipynb

## Files

| File | Purpose |
|---|---|
| `Student_Marks_Prediction.ipynb` | Main notebook: all 7 project steps, run top to bottom |
| `student_marks.csv` | Dataset (1,020 rows, synthetic, with missing values, duplicates and outliers on purpose) |
| `generate_dataset.py` | Script that creates `student_marks.csv` (fixed seed, always the same data) |
| `app.py` | Streamlit UI: enter student details → predicted marks, grade, input summary |
| `model/` | Saved `model.pkl`, `scaler.pkl`, `artifacts.pkl` (created by the notebook) |
| `requirements.txt` | Python packages for running locally |

## Run in Google Colab

1. Open https://colab.research.google.com → **Upload** → choose `Student_Marks_Prediction.ipynb`.
2. (Optional) In the left **Files** panel, upload `student_marks.csv`. If you skip this, the notebook regenerates the identical dataset itself.
3. Run the cells one by one (Shift+Enter), or **Runtime → Run all**.
4. In section 6, the launch cell prints a `https://....trycloudflare.com` link. Open it to use the Streamlit app.
5. The last cell downloads a zip with the trained model and app.

## Run locally

```
pip install -r requirements.txt
jupyter notebook Student_Marks_Prediction.ipynb   # run all cells, creates model/
streamlit run app.py
```

## Project steps → notebook sections

1. Data Collection – section 1 (load data, column descriptions)
2. Data Preprocessing – section 2 (missing values, duplicates, outliers, one-hot encoding, correlation-based feature selection)
3. Model Building – section 3 (80/20 split, StandardScaler, LinearRegression)
4. Model Evaluation – section 4 (MAE, MSE, RMSE, R², actual-vs-predicted and residual plots)
5. Save Model – section 5 (joblib)
6. Streamlit UI – section 6 (`app.py`)
7. Interpretation – section 7 (coefficients, R² meaning, accuracy, limitations)

## Results (test set)

R² = 0.823, MAE = 3.55 marks, RMSE = 4.47 marks.

## Deploy on Streamlit Community Cloud (public link)

1. Push this folder to a public GitHub repo (it must include `app.py`, `requirements.txt` and the `model/` folder).
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app** → **Deploy a public app from GitHub**.
3. Pick the repo, branch `main`, main file `app.py`, then **Deploy**.
4. After about a minute you get a permanent link. This project is deployed at https://students-mark-prediction.streamlit.app/

`requirements.txt` pins `scikit-learn==1.9.1`, the version `model/model.pkl` was saved with. If you retrain in Colab and replace the `model/` files, change that pin to Colab's version (`import sklearn; sklearn.__version__`).
