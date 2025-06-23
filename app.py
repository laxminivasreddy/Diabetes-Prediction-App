# importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import joblib
from tensorflow import keras
import streamlit as st

# Set Streamlit page configuration (fix layout string)
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

# Title and description
st.title("Diabetes Prediction App 🚀")
st.markdown("Enter the following details to predict the likelihood of diabetes:")

# Load Model and Scaler
try:
    model = tf.keras.models.load_model("Diabetes_model.h5")
    scaler = joblib.load("Scaler.pkl")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Input fields
pregnancies = st.number_input("Number of pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose level", min_value=0)
bloodpressure = st.number_input("Blood pressure", min_value=0)
skinthickness = st.number_input("Skin thickness", min_value=0)
insulin = st.number_input("Insulin level", min_value=0)
bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0)
diabetespedigreefunction = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Predict Button
if st.button("🔍 Predict Diabetes"):
    input_data = np.array([[pregnancies, glucose, bloodpressure, skinthickness,
                            insulin, bmi, diabetespedigreefunction, age]])
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0][0]
        result = "🟢 Not Diabetic" if prediction < 0.5 else "🔴 Diabetic"
        st.subheader("Prediction Result:")
        st.success(result)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
