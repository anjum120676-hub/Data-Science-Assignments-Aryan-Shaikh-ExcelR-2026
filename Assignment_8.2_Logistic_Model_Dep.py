#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import pickle
import streamlit as st
import os


# In[4]:


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load model & scaler
try:
    model = pickle.load(open(os.path.join(script_dir, "model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(script_dir, "scaler.pkl"), "rb"))
except FileNotFoundError as e:
    st.error(f"Error: {e}. Make sure model.pkl and scaler.pkl files exist.")
    st.stop()

st.title("Diabetes Prediction App")

# User Inputs   
preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=0)
bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=0)
skin = st.number_input("Skin Thickness", min_value=0, max_value=99, value=0)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.0)
age = st.number_input("Age", min_value=1, max_value=120, value=20)

# Arrange input for model
input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

# Scale the input
scaled_data = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(scaled_data)[0]
    if prediction == 1:
        st.error("The model predicts: Diabetes Positive")
    else:
        st.success("The model predicts: Diabetes Negative")


# In[ ]:




