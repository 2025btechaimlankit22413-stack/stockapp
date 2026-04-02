import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")

# ---------------- TITLE ----------------
st.title('📈 Stock Trend Prediction')

# ---------------- INPUT ----------------
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

start = '2010-01-01'
end = '2026-03-26'

# ---------------- DATA FETCH ----------------
df = yf.download(user_input, start=start, end=end, progress=False)

# ---------------- DATA CHECK ----------------
if df is None or df.empty or 'Close' not in df.columns:
    st.error("❌ No valid data found. Check ticker.")
    st.stop()

# ---------------- CLEAN DATA ----------------
df = df[['Close']].copy()
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df.dropna(inplace=True)

if len(df) < 10:
    st.error("❌ Not enough data.")
    st.stop()

# ---------------- DATA INFO ----------------
st.subheader('📊 Data Summary')
st.write(df.describe())

# ---------------- CHART ----------------
st.subheader('📉 Closing Price')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.title("Closing Price")
st.pyplot(fig)

# ---------------- MOVING AVERAGES ----------------
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

st.subheader('📊 Moving Averages')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Close')
plt.plot(ma100, label='MA100')
plt.plot(ma200, label='MA200')
plt.legend()
st.pyplot(fig)

# ---------------- MODEL DATA ----------------
df['Prediction'] = df['Close'].shift(-1)
df.dropna(inplace=True)

X = df[['Close']].values.astype(float)
y = df['Prediction'].values.astype(float)

# ---------------- TRAIN MODEL ----------------
model = LinearRegression()
model.fit(X, y)

# ---------------- PREDICTION ----------------
y_pred = model.predict(X)

# ---------------- GRAPH ----------------
st.subheader('📈 Prediction vs Actual')

fig2 = plt.figure(figsize=(12,6))
plt.plot(y, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
st.pyplot(fig2)

# ---------------- CURRENT PRICE ----------------
current_price = float(df['Close'].iloc[-1])
st.subheader("📍 Current Price")
st.success(f"${current_price:.2f}")

# ---------------- METRICS ----------------
if len(df) >= 2:
    previous_price = float(df['Close'].iloc[-2])
    change = current_price - previous_price
    percent = (change / previous_price) * 100
else:
    change = 0
    percent = 0

col1, col2 = st.columns(2)

with col1:
    st.metric("Price", f"${current_price:.2f}")

with col2:
    st.metric("Change", f"{change:.2f}", f"{percent:.2f}%")

# ---------------- SAFE PREDICTION ----------------
st.subheader("🔮 Next Day Prediction")

try:
    next_day = model.predict(np.array([[current_price]]))[0]
    st.success(f"Predicted Next Day Price: ${float(next_day):.2f}")
except:
    st.error("Prediction failed")
