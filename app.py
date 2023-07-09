import pandas_datareader as data
from keras.models import load_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

start = '2010-01-01'
end = '2023-01-01'

st.title('Stock Trend Visualization by [@manavdhanani](https:man9099.github.io)')

user_input = st.text_input('Enter any Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)

# Describing Data
st.subheader(f'Data from {start} - {end}')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time chart')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Close'].values, 'b')  # Corrected column name
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price')
ax.set_title(f'{user_input} Stock Closing Prices')
ax.grid(True)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Close'].values, 'b')
ax.plot(ma100.values, 'g')
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price')
ax.set_title(f'{user_input} Stock Closing Prices')
ax.grid(True)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Close'].values, 'b')
ax.plot(ma200.values, 'r')
ax.plot(ma100.values, 'g')
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price')
ax.set_title(f'{user_input} Stock Closing Prices')
ax.grid(True)
st.pyplot(fig)




#load my model
model = load_model('keras_model.h5')
