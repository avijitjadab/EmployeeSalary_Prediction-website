import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("salary_model.pkl")

# You can optionally load scalers if saved separately:
# age_scaler = joblib.load("age_scaler.pkl")
# exp_scaler = joblib.load("exp_scaler.pkl")

# Hardcoded encodings (must match training)
degree_map = {'Bachelors': 0, 'Masters': 1, 'PhD': 2}
job_title_map = {'Data Scientist': 0, 'Software Engineer': 1, 'Manager': 2, 'HR': 3}
gender_map = {'Male': 1, 'Female': 0}

# Streamlit UI
st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction App")
st.markdown("Fill the employee's details to estimate their salary.")

# Input fields
age = st.slider("Age", 18, 65, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
degree = st.selectbox("Education Level", list(degree_map.keys()))
job_title = st.selectbox("Job Title", list(job_title_map.keys()))
experience = st.slider("Years of Experience", 0, 40, 5)

# Button
if st.button("Predict Salary"):
    # Encode categorical features
    gender_encoded = gender_map[gender]
    degree_encoded = degree_map[degree]
    job_title_encoded = job_title_map[job_title]

    # Standard scaling (fit fresh, or load saved scalers)
    scaler = StandardScaler()
    scaled_inputs = scaler.fit_transform([[age, experience]])
    age_scaled, exp_scaled = scaled_inputs[0]

    # Final input for model
    input_data = np.array([[age_scaled, gender_encoded, degree_encoded, job_title_encoded, exp_scaled]])
    predicted_salary = model.predict(input_data)[0]

    st.success(f"💰 Predicted Salary: ₹ {predicted_salary:,.2f}")
