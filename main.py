import streamlit as st
import joblib
import numpy as np

# Load the trained model
try:
    model = joblib.load("random_forest_model.pkl")  # Change if using a different model
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title("Coronary Artery Disease Detection")
st.write("Enter patient details to predict the likelihood of CAD.")

# Input fields for the 10 features
age = st.number_input("Age", min_value=1, max_value=120, value=50)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
htn = st.selectbox("Hypertension (HTN)", ["Yes", "No"])
bp = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
typical_cp = st.selectbox("Typical Chest Pain", ["Yes", "No"])
tinversion = st.selectbox("T Inversion", ["Yes", "No"])
fbs = st.number_input("Fasting Blood Sugar (FBS)", min_value=50, max_value=300, value=100)
cr = st.number_input("Creatinine (CR)", min_value=0.1, max_value=5.0, value=1.0)
k = st.number_input("Potassium (K)", min_value=2.0, max_value=7.0, value=4.0)
region_rwma = st.selectbox("Regional Wall Motion Abnormalities (RWMA)", ["Yes", "No"])

# Convert categorical values to numerical format
htn = 1 if htn == "Yes" else 0
typical_cp = 1 if typical_cp == "Yes" else 0
tinversion = 1 if tinversion == "Yes" else 0
region_rwma = 1 if region_rwma == "Yes" else 0

# Create feature array
features = np.array([age, bmi, htn, bp, typical_cp, tinversion, fbs, cr, k, region_rwma]).reshape(1, -1)

# Debugging: Show input features
st.write("Feature Array Sent to Model:", features)

# Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(features)
        st.write("Raw Prediction Output:", prediction)

        # If the model supports probability prediction, show it
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)
            st.write("Prediction Probability:", proba)

        result = "Likely CAD" if prediction[0] == 1 else "Normal"
        st.write(f"**Prediction:** {result}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
