import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV


model = joblib.load("best_model.joblib")
encoder = joblib.load("encoder.joblib")

st.set_page_config(
    page_title="EcoType: Forest Cover Type Prediction Using Machine Learning",
    layout="wide"
)

st.title("EcoType: Forest Cover Type Prediction Using Machine Learning")

st.subheader("Environmental Data & Geospatial Predictive Modeling")

st.write("Enter the details below:")

Elevation = st.number_input("Elevation")
Aspect = st.number_input("Aspect")
Slope = st.number_input("Slope")
Horizontal_Distance_To_Hydrology = st.number_input("Horizontal Distance To Hydrology")
Vertical_Distance_To_Hydrology = st.number_input("Vertical Distance To Hydrology")
Horizontal_Distance_To_Roadways = st.number_input("Horizontal Distance To Roadways")
Hillshade_9am = st.number_input("Hillshade 9am")
Hillshade_Noon = st.number_input("Hillshade Noon")
Hillshade_3pm = st.number_input("Hillshade 3pm")
Horizontal_Distance_To_Fire_Points = st.number_input("Horizontal Distance To Fire Points")
Wilderness_Area = st.selectbox("Wilderness Area", options=list(range(1, 5)))
Soil_Type = st.selectbox("Soil Type", options=list(range(1, 41)))




if st.button("Predict"):

    # Convert input into array
    input_data = np.array([[Elevation, Aspect, Slope,
                            Horizontal_Distance_To_Hydrology,
                            Vertical_Distance_To_Hydrology,
                            Horizontal_Distance_To_Roadways,
                            Hillshade_9am, Hillshade_Noon, Hillshade_3pm,
                            Horizontal_Distance_To_Fire_Points,
                            Wilderness_Area, Soil_Type]])

    # Prediction
    prediction = model.predict(input_data)
    
    label = encoder.inverse_transform(prediction)

    st.success(f"Predicted Cover Type: {label[0]}")
