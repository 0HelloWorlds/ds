import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Load the saved artifacts
scaler = joblib.load('scaler.pkl')
model = joblib.load('mlp_model.pkl')

# Define the features in order
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
            'total_bedrooms', 'population', 'households', 'median_income', 
            'ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 
            'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']

st.title("California House Price Predictor")

# Create inputs
inputs = []
for feat in features:
    val = st.number_input(f"Enter {feat}", value=0.0)
    inputs.append(val)

if st.button("Predict"):
    # Transform input and predict
    X_input = np.array([inputs])
    X_scaled = scaler.transform(X_input)
    prediction = model.predict(X_scaled)
    
    st.success(f"The predicted house value is: ${prediction[0]:,.2f}")
