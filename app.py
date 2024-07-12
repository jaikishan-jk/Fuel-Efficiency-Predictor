import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the saved model and encoders

def load_model():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def load_encoders():
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return encoders

# Load data for input options
data = pd.read_csv('Fuel Efficiency.csv')

# Rename transmission and fuel type columns
data = data.replace({'Transmission' : {'AM8':'AM', 'AS10': 'AS', 'A8':'A', 'A9':'A', 'AM7':'AM', 'AS8':'AS', 'M6':'M','AS6':'AS', 'AS9':'AS', 'A10':'A', 'A6':'A', 'M5':'M', 'M7':'M', 'AV7':'AV', 'AV1':'AV', 'AM6':'AM', 'AS7':'AS', 'AV8':'AV', 'AV6':'AV', 'AV10':'AV', 'AS5':'AS', 'A7':'A','AM9':'AM','A5':'A','A4':'A','AS4':'AS','AM5':'AM','A3':'A','M4':'M'}})
data = data.replace({'Fuel type' : {'D':'Diesel', 'E': 'Hybrid', 'X':'CNG', 'Z':'Gasoline'}})

label_encoders = load_encoders()

# Streamlit UI
st.title('Fuel Efficiency Predictor')

# Input form
st.sidebar.header('Input Parameters')

# Input fields
vehicle_class = st.sidebar.selectbox('Vehicle class', data['Vehicle class'].unique())
engine_size = st.sidebar.slider('Engine size (L)', min_value=1.0, max_value=8.0, value=None, step=0.1)
cylinders = st.sidebar.slider('Cylinders', min_value=3, max_value=16, value=None, step=1)
transmission = st.sidebar.selectbox('Transmission', data['Transmission'].unique())
fuel_type = st.sidebar.selectbox('Fuel type', data['Fuel type'].unique())
co2_rating = st.sidebar.slider('CO2 rating', min_value=1, max_value=10, value=None, step=1)

# Predict function
def predict(vehicle_class, engine_size, cylinders, transmission, fuel_type, co2_rating):
    input_data = pd.DataFrame({
        'Vehicle class': [vehicle_class],
        'Engine size (L)': [engine_size],
        'Cylinders': [cylinders],
        'Transmission': [transmission],
        'Fuel type': [fuel_type],
        'CO2 rating': [co2_rating]
    })

    # Encode categorical variables
    for column in label_encoders:
        input_data[column] = label_encoders[column].transform(input_data[column])

    # Load the model
    model = load_model()

    # Predict
    prediction = model.predict(input_data)
    return prediction[0]

# Prediction
if st.sidebar.button('Predict'):
    prediction = predict(vehicle_class, engine_size, cylinders, transmission, fuel_type, co2_rating)
    st.subheader('Prediction Result')
    st.write(f'Predicted fuel consumption: {prediction:.2f} L/100 km')


