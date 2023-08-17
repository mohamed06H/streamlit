import os

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Header 
st.write("""
# Titanic Survival Prediction App
This app predicts if you would survive to titanic or not.
""")

# Sidebar
st.sidebar.header('User Input Features')


def user_input_features():
    embarked = st.sidebar.selectbox('Embarkation port', ('C', 'S', 'Q'))
    sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    pclass = st.sidebar.selectbox('Passenger Class', (1, 2, 3))
    age = st.sidebar.slider('Age', 0, 99, 25)
    fare = st.sidebar.slider('Fare', 0, 600, 33)
    data = {'embarked': embarked,
            'sex': sex,
            'pclass': pclass,
            'age': age,
            'fare': fare}

    features = pd.DataFrame(data, index=[0])
    return features


# Input data from sidebar
input_df = user_input_features()

# Input data in a subheader
st.subheader('User Input features')
st.write(input_df)

# Load saved model
pickle_file = open('./model/titanic.pkl')
model = pickle.load(pickle_file, 'rb'))

# Make prediction
prediction = model.predict(input_df)
prediction = int(prediction[0])
prediction_proba = model.predict_proba(input_df)

# Predictions 
st.subheader('Prediction')
outcome = np.array(['Survived', 'Did not survived !'])
st.write(outcome[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
