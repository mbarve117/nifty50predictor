import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import keras
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

start = "2010-01-01"
end = "2019-12-31"

st.title("Stock Trend Prediction with LSTM")

user_input = st.text_input("Enter Stock Ticker", "AAPL") # This is the placeholder ticker
df = yf.download(user_input, start, end)

# Descriptions
st.subheader("Data from 2010 - 2019")
st.write(df.describe())

# Visualizations
st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize=(12,6))
plt.plot(df["Close"])
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100MA & 200MA")
ma100 = df["Close"].rolling(100).mean()
ma200 = df["Close"].rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df["Close"], 'b')
st.pyplot(fig)

# Data
train_data = pd.DataFrame(df["Close"][0:int(len(df)*0.70)])
test_data = pd.DataFrame(df["Close"][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
train_array = scaler.fit_transform(train_data)

x_train = []
y_train = []

for i in range(100, train_array.shape[0]):
    x_train.append(train_array[i-100:i])
    y_train.append(train_array[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

model = keras.saving.load_model("keras_model.keras")

past_100_days = train_data.tail(100)
final_df = pd.concat([past_100_days, test_data], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_pred = model.predict(x_test)
scaler.scale_

scale_factor = 1/0.02123255
y_test = y_test * scale_factor
y_pred = y_pred * scale_factor

st.subheader("Predictions vs Original")
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = "Original Price")
plt.plot(y_pred, 'r', label = "Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)
