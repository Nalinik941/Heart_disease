# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 12:17:36 2025

@author: Dell
"""

#importing reuired libraries
import streamlit as st
import joblib

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load data
data = pd.read_excel('patient_dataset.xlsx')





# Encode categorical features
encoder = LabelEncoder()
data['smoking_status'] = encoder.fit_transform(data['smoking_status'])
data['residence_type'] = encoder.fit_transform(data['residence_type'])

# Train KMeans clustering
features = ['age', 'gender', 'chest_pain_type', 'blood_pressure', 'cholesterol',
            'max_heart_rate', 'exercise_angina', 'plasma_glucose', 'skin_thickness',
            'insulin', 'bmi', 'diabetes_pedigree', 'hypertension', 'heart_disease',
            'residence_type', 'smoking_status']
X = data[features]
# Impute missing values
imputer = SimpleImputer(strategy='mean')  # Use 'mean', 'median', or 'most_frequent' as needed
X_imputed = imputer.fit_transform(X)


kmeans = KMeans(n_clusters=4, random_state=42)
data['cluster'] = kmeans.fit_predict(X_imputed)


# Train Logistic Regression model
X = pd.DataFrame(X_imputed, columns=features)
y = data['cluster']
model = LogisticRegression()
model.fit(X, y)


# Streamlit UI
st.title('💖 Heart Disease Prediction')
st.write("Enter patient details to predict heart disease cluster:")

# Create input fields
age = st.number_input('Age', min_value=0, max_value=120, value=45)
gender = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
chest_pain_type = st.number_input('Chest Pain Type', min_value=0, max_value=3, value=1)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=300, value=120)
cholesterol = st.number_input('Cholesterol', min_value=0, max_value=600, value=200)
max_heart_rate = st.number_input('Max Heart Rate', min_value=0, max_value=250, value=150)
exercise_angina = st.selectbox('Exercise Angina', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
plasma_glucose = st.number_input('Plasma Glucose', min_value=0, max_value=300, value=100)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.number_input('Insulin', min_value=0, max_value=500, value=80)
bmi = st.number_input('BMI', min_value=0.0, max_value=60.0, value=25.5)
diabetes_pedigree = st.number_input('Diabetes Pedigree', min_value=0.0, max_value=2.5, value=0.5)
hypertension = st.selectbox('Hypertension', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
heart_disease = st.selectbox('Heart Disease', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
residence_type = st.selectbox('Residence Type', options=[0, 1], format_func=lambda x: 'Urban' if x == 1 else 'Rural')
smoking_status = st.number_input('Smoking Status', min_value=0, max_value=3, value=1)

# Prediction Button
if st.button('Predict'):
    input_data = pd.DataFrame([[age, gender, chest_pain_type, blood_pressure, cholesterol,
                                max_heart_rate, exercise_angina, plasma_glucose, skin_thickness,
                                insulin, bmi, diabetes_pedigree, hypertension, heart_disease,
                                residence_type, smoking_status]],
                              columns=features)
    
    # Predict cluster using Logistic Regression
    predicted_cluster = model.predict(input_data)[0]

    # Display Results
    st.success(f"Predicted Cluster: {predicted_cluster}")
    if predicted_cluster == 0:
        st.write('💚 Low Risk')
    elif predicted_cluster == 1:
        st.write('💛 Moderate Risk')
    elif predicted_cluster == 2:
        st.write('🧡 High Risk')
    else:
        st.write('❤️ Critical Risk')

# About Section
st.sidebar.title('About')
st.sidebar.info('This is a heart disease prediction model using KMeans clustering and Logistic Regression.')
