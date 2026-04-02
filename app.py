import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")

# ---------------- INPUT ----------------
st.title('📈 Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

start = '2010-01-01'
end = '2026-03-26'

# ---------------- DATA FETCH ----------------
df = yf.download(user_input, start=start, end=end, progress=False)

# ---------------- DATA CHECK ----------------


# ---------------- CLEAN DATA ----------------
df = df[['Close']].copy()
df.dropna(inplace=True)

if len(df) < 5:
    st.error("⚠️ Not enough data.")
    st.stop()

# ---------------- DATA ----------------
st.subheader('Data Summary')
st.write(df.describe())

# ---------------- CHART ----------------
st.subheader('Closing Price')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
st.pyplot(fig)

# ---------------- MA ----------------
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Close')
plt.plot(ma100, label='MA100')
plt.plot(ma200, label='MA200')
plt.legend()
st.pyplot(fig)

# ---------------- MODEL DATA ----------------
df['Prediction'] = df['Close'].shift(-1)
df.dropna(inplace=True)

X = np.array(df[['Close']], dtype=float)
y = np.array(df['Prediction'], dtype=float)

# ---------------- TRAIN ----------------
model = LinearRegression()
model.fit(X, y)

# ---------------- PREDICT ----------------
y_pred = model.predict(X)

# ---------------- CURRENT PRICE ----------------
current_price = float(df['Close'].iloc[-1])

# ---------------- GRAPH ----------------
st.subheader('Prediction vs Actual')

fig2 = plt.figure(figsize=(12,6))
plt.plot(y, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.axhline(y=current_price, linestyle='--', label='Current')
plt.legend()
st.pyplot(fig2)

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

# ---------------- SIDEBAR ----------------
st.sidebar.title("Settings")
future_days = st.sidebar.slider("Future Days", 5, 50, 10)

# ---------------- FUTURE PREDICTION ----------------
st.subheader('🔮 Future Prediction')

last_price = float(df['Close'].iloc[-1])
future_predictions = []

current_input = last_price

for i in range(future_days):
    # SAFE CONVERSION
    if not np.isfinite(current_input):
        break

    input_array = np.array([[float(current_input)]], dtype=float)

    pred = model.predict(input_array)

    if not np.isfinite(pred[0]):
        break

    pred_value = float(pred[0])

    future_predictions.append(pred_value)
    current_input = pred_value

# ---------------- FUTURE DATA ----------------
from datetime import datetime

dates = pd.bdate_range(start=datetime.now(), periods=len(future_predictions))

future_df = pd.DataFrame({
    "Date": dates,
    "Predicted Price": future_predictions
})

st.write(future_df)

# ---------------- FUTURE GRAPH ----------------
fig3 = plt.figure(figsize=(12,6))
plt.plot(future_predictions, label='Future')
plt.legend()
st.pyplot(fig3)
