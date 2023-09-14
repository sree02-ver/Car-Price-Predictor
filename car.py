import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the pre-trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.header('Car Price Predictor')

# User inputs
year = st.number_input('Make year')
kms = st.number_input('KMs driven')
fuel = st.selectbox('Fuel Type', ('Diesel', 'Petrol'))
seller = st.selectbox('Seller Type', ('Individual', 'Dealer'))
transmission = st.selectbox('Transmission', ('Manual', 'Automatic'))
owner = st.selectbox('Owner', ('First Owner', 'Second Owner', 'Third Owner'))
mileage = st.number_input('Mileage')
engine = st.number_input('Engine')
power = st.number_input('Max Power')
seats = st.number_input('Seats')
brand = st.selectbox('Brand', ('Maruti', 'Hyundai', 'Mahindra', 'Tata', 'Ford', 'Honda', 'Toyota', 'Renault', 'Chevrolet', 'Volkswagen'))

if st.button('Predict Price'):
    # Create a user input DataFrame
    user_input = np.array([[year, kms, fuel, seller, transmission, owner, mileage, engine, power, seats, brand]])
    user_input_df = pd.DataFrame(user_input,
                                 columns=['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner',
                                          'mileage', 'engine', 'max_power', 'seats', 'brand'])

    # Make predictions
    predicted_price = pipe.predict(user_input_df)

    # Display the predicted price
    st.title("Predicted Price: Rs {:.2f}".format(predicted_price[0]))
