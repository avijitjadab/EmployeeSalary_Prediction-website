# app.py
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Predictor")

# MODEL_PATH should match the filename in your repo (put model in repo root)
MODEL_PATH = "salary_model_pipeline_colab.pkl"  # or "salary_model.pkl"

@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)

pipe = load_model(MODEL_PATH)

# UI inputs
age = st.slider("Age", 18, 65, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", sorted(["Bachelors", "Masters", "PhD"]))
job_title = st.text_input("Job Title", "Software Engineer")
experience = st.slider("Years of Experience", 0, 40, 3)

if st.button("Predict Salary"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Education Level": education,
        "Job Title": job_title,
        "Years of Experience": experience
    }])
    # If you grouped rare jobs into "Other" during training, map unknowns:
    # known_titles = [...]  # optional: replace with your top titles
    # input_df["Job Title"] = input_df["Job Title"].apply(lambda t: t if t in known_titles else "Other")

    try:
        pred = pipe.predict(input_df)[0]
        st.success(f"💰 Predicted Salary: ₹ {pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write("Possible causes: model filename mismatch, different feature names, unseen preprocessing step.")

