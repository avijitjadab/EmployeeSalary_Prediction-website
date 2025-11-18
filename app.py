# app.py
import os
import streamlit as st
import joblib
import pandas as pd
import difflib
import traceback

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Predictor")

# ------------------ Model loading (joblib, fallback to cloudpickle) ------------------
MODEL_PATH = "salary_model.pkl"  # change if your model filename is different

@st.cache_resource(show_spinner=False)
def load_model(path):
    # Try joblib first
    try:
        return joblib.load(path)
    except Exception as e_joblib:
        # Try cloudpickle as fallback
        try:
            import cloudpickle
        except Exception:
            raise RuntimeError(
                "Failed to load model with joblib. cloudpickle not installed. "
                "Install cloudpickle or re-save your model."
            ) from e_joblib
        try:
            with open(path, "rb") as f:
                return cloudpickle.load(f)
        except Exception as e_cloud:
            # Raise combined info for debugging
            raise RuntimeError(
                "Failed to load model with joblib and cloudpickle. "
                f"joblib error: {e_joblib}\ncloudpickle error: {e_cloud}"
            )

# Check model file exists early and display helpful message
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Place the .pkl file in the app folder and restart.")
    st.stop()

try:
    pipe = load_model(MODEL_PATH)
except Exception as e:
    st.error("Error loading model. See details below.")
    st.text(str(e))
    st.write("Traceback:")
    st.text(traceback.format_exc())
    st.stop()

st.success(f"Loaded model: {os.path.basename(MODEL_PATH)}")

# ------------------ Basic UI inputs ------------------
age = st.slider("Age", 18, 65, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", sorted(["Bachelors", "Masters", "PhD"]))
experience = st.slider("Years of Experience", 0, 40, 3)

# ------------------ Job title extraction & handling ------------------
def extract_known_job_titles(pipeline):
    """
    Try to read the OneHotEncoder categories from the pipeline's preprocessor.
    Returns a list of known job titles (or empty list if not found).
    """
    try:
        pre = pipeline.named_steps.get("preprocessor", None) or pipeline[0]
    except Exception:
        return []

    known = []
    try:
        transformers = getattr(pre, "transformers_", [])
        for tname, transformer, cols in transformers:
            # transformer might be a Pipeline containing a OneHotEncoder named "onehot"
            ohe = None
            if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
                ohe = transformer.named_steps["onehot"]
            elif hasattr(transformer, "categories_"):
                ohe = transformer
            if ohe is not None and hasattr(ohe, "categories_"):
                cats = list(ohe.categories_)
                col_names = list(cols) if isinstance(cols, (list, tuple)) else []
                # If number of cat arrays equals number of cat columns, find Job Title
                if len(col_names) == len(cats):
                    for i, colname in enumerate(col_names):
                        if colname and colname.lower().strip() in ["job title", "job_title", "jobtitle", "title"]:
                            known = [str(x) for x in cats[i]]
                            return known
                else:
                    # heuristic: if there are 3 categorical arrays, assume last is Job Title
                    if len(cats) == 3:
                        known = [str(x) for x in cats[2]]
                        return known
    except Exception:
        return []

    return known

known_job_titles = extract_known_job_titles(pipe)
if not known_job_titles:
    # fallback defaults (you can edit these)
    known_job_titles = ["Software Engineer", "Data Scientist", "Manager", "HR", "Other"]

# fuzzy mapping helper
def map_to_known(title, known_list, cutoff=0.6):
    t = (title or "").strip()
    if t == "":
        return t
    t_low = t.lower()
    for k in known_list:
        if t_low == str(k).strip().lower():
            return k
    lower_known = [str(k).strip().lower() for k in known_list]
    matches = difflib.get_close_matches(t_low, lower_known, n=1, cutoff=cutoff)
    if matches:
        idx = lower_known.index(matches[0])
        return known_list[idx]
    # fallback to Other if present
    if "Other" in known_list:
        return "Other"
    return title

# UI for job title: dropdown + "Other (type below)"
st.write("## Job Title (choose or type)")
options = known_job_titles + ["Other (type below)"]
job_choice = st.selectbox("Select Job Title", options, index=0 if len(known_job_titles) else 0)
if job_choice == "Other (type below)":
    job_text = st.text_input("Type custom job title", "Other")
    job_title_input = job_text.strip() if job_text.strip() != "" else "Other"
else:
    job_title_input = job_choice

# Map to training category (fuzzy)
job_title_mapped = map_to_known(job_title_input, known_job_titles, cutoff=0.6)

st.write("Using job title for model:", f"**{job_title_mapped}**")

# ------------------ Build input and predict ------------------
input_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Education Level": education,
    "Job Title": job_title_mapped,
    "Years of Experience": experience
}])

# (Optional) Show input to help debug
with st.expander("Show input passed to model (click to expand)"):
    st.write(input_df)

if st.button("Predict Salary"):
    try:
        pred = pipe.predict(input_df)[0]
        st.success(f"💰 Predicted Salary: ₹ {pred:,.2f}")
    except Exception as ex:
        st.error("Prediction failed. See error below:")
        st.text(str(ex))
        st.write("Traceback:")
        st.text(traceback.format_exc())
        st.write(
            "Notes:\n"
            "- Ensure your trained pipeline expects these exact column names: "
            "`Age`, `Gender`, `Education Level`, `Job Title`, `Years of Experience`.\n"
            "- If your model used different category names, re-export the pipeline or update this app."
        )
