import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('salary_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'salary_model.pkl' not found. Please upload the model file.")
        return None

# Initialize scalers and encoders (you'll need to fit these with your original training data)
@st.cache_resource
def initialize_preprocessors():
    # These should ideally be saved from your training process
    # For now, we'll create new ones - you should replace with your actual fitted scalers
    age_scaler = StandardScaler()
    experience_scaler = StandardScaler()
    
    # Fit with approximate ranges (replace with your actual training data ranges)
    age_scaler.fit([[18], [65]])  # Approximate age range
    experience_scaler.fit([[0], [40]])  # Approximate experience range
    
    return age_scaler, experience_scaler

def main():
    st.title("💰 Employee Salary Prediction")
    st.markdown("---")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Initialize preprocessors
    age_scaler, experience_scaler = initialize_preprocessors()
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        age = st.slider("Age", min_value=18, max_value=65, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
    with col2:
        st.subheader("Professional Information")
        degree = st.selectbox("Degree", [
            "Bachelor's", "Master's", "PhD", "High School", "Associate"
        ])
        job_title = st.selectbox("Job Title", [
            "Software Engineer", "Data Scientist", "Manager", "Analyst", 
            "Director", "Senior Engineer", "Consultant", "Developer"
        ])
    
    experience_years = st.slider("Years of Experience", min_value=0, max_value=40, value=5)
    
    # Prediction button
    if st.button("Predict Salary", type="primary"):
        try:
            # Preprocess inputs
            # Scale age and experience
            age_scaled = age_scaler.transform([[age]])[0][0]
            experience_scaled = experience_scaler.transform([[experience_years]])[0][0]
            
            # Encode categorical variables
            # Gender encoding (adjust based on your training data)
            gender_map = {"Male": 0, "Female": 1, "Other": 2}
            gender_encoded = gender_map.get(gender, 0)
            
            # Degree encoding (adjust based on your training data)
            degree_map = {
                "High School": 0,
                "Associate": 1,
                "Bachelor's": 2,
                "Master's": 3,
                "PhD": 4
            }
            degree_encoded = degree_map.get(degree, 2)
            
            # Job title encoding (adjust based on your training data)
            job_title_map = {
                "Analyst": 0,
                "Consultant": 1,
                "Data Scientist": 2,
                "Developer": 3,
                "Director": 4,
                "Manager": 5,
                "Senior Engineer": 6,
                "Software Engineer": 7
            }
            job_title_encoded = job_title_map.get(job_title, 7)
            
            # Create feature array in the correct order
            features = np.array([[
                age_scaled,           # Age_Scaled
                gender_encoded,       # Gender_Encode
                degree_encoded,       # Degree_Encode
                job_title_encoded,    # Job_Title_Encode
                experience_scaled     # Experience_years_Scaled
            ]])
            
            # Make prediction
            predicted_salary = model.predict(features)[0]
            
            # Display results
            st.success(f"### Predicted Annual Salary: ${predicted_salary:,.2f}")
            
            # Show input summary
            st.markdown("---")
            st.subheader("Input Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Age", age)
                st.metric("Experience", f"{experience_years} years")
            
            with col2:
                st.metric("Gender", gender)
                st.metric("Degree", degree)
            
            with col3:
                st.metric("Job Title", job_title)
                st.metric("Predicted Salary", f"${predicted_salary:,.0f}")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please check that your model file is compatible and all preprocessing steps are correct.")

# IMPORTANT NOTE for the user
st.sidebar.markdown("""
## ⚠️ Important Setup Instructions

**To fix the scaling issue completely, you need to:**

1. **Save your original scalers** from training:
```python
# During training, save your scalers
pickle.dump(age_scaler, open('age_scaler.pkl', 'wb'))
pickle.dump(experience_scaler, open('experience_scaler.pkl', 'wb'))
```

2. **Load the saved scalers** in this app instead of creating new ones

3. **Update the encoding mappings** to match your training data exactly

The current code creates approximate scalers, but for best results, use your original fitted scalers!
""")

if __name__ == "__main__":
    main()
