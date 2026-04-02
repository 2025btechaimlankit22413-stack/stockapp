import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf 
import streamlit as st

from sklearn.linear_model import LinearRegression

start = '2010-01-01'
end = '2026-03-26'

st.title('📈 Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)

# ---------------- DATA ----------------
st.subheader('Data from 2010-2026')
st.write(df.describe())

# ---------------- CHARTS ----------------
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

# ---------------- PREPARE DATA ----------------
df = df[['Close']]
df['Prediction'] = df['Close'].shift(-1)

X = np.array(df[['Close']][:-1])
y = np.array(df['Prediction'][:-1])

# ---------------- TRAIN MODEL ----------------
model = LinearRegression()
model.fit(X, y)

# ---------------- TEST ----------------
y_predicted = model.predict(X)

# ---------------- LIVE PRICE ----------------
df_live = yf.download(user_input, period="5y")

if not df_live.empty:
    current_price = float(df_live['Close'].iloc[-1])
else:
    current_price = 0

# ---------------- GRAPH ----------------
st.subheader('Predictions vs Original + Live')

fig2 = plt.figure(figsize=(12,6))

plt.plot(y, label='Original', linewidth=2)
plt.plot(y_predicted, label='Predicted', linewidth=2)

plt.axhline(y=current_price, linestyle='--', label='Live Price')

plt.grid(alpha=0.3)
plt.legend()
st.pyplot(fig2)

# ---------------- LIVE PRICE DISPLAY ----------------
if not df_live.empty:
    st.subheader("📍 Current Live Price")
    st.success(f"₹ {current_price:.2f}")
else:
    st.error("Live price not available ❌")

# ---------------- SIDEBAR ----------------
st.sidebar.title("FILTER 📈 ")
ticker = st.sidebar.text_input("Enter Stock", "AAPL")
days = st.sidebar.slider("Future Days", 10, 100, 10)

st.line_chart(df['Close'])

# ---------------- METRICS ----------------
if len(df_live) > 1:
    previous_price = float(df_live['Close'].iloc[-2])
    change = current_price - previous_price
    percent = (change / previous_price) * 100
else:
    change = 0
    percent = 0

st.set_page_config(layout="wide")

col1, col2 = st.columns(2)

with col1:
    st.metric("Price", f"${current_price:.2f}")

with col2:
    st.metric("Change", f"{change:.2f}", f"{percent:.2f}%")

# ---------------- FUTURE PREDICTION ----------------
st.subheader('🔮 Future Predictions')

future_days = days
last_price = df['Close'].iloc[-1]

future_predictions = []

current_input = last_price

for i in range(future_days):
    pred = model.predict([[current_input]])
    future_predictions.append(pred[0])
    current_input = pred[0]

future_predictions = np.array(future_predictions)

from datetime import datetime
dates = pd.bdate_range(start=datetime.now(), periods=future_days)

future_df = pd.DataFrame({
    "Date": dates,
    "Predicted Price": future_predictions
})

st.subheader(f"📅 Next {future_days} Days Prediction")
st.write(future_df)

# ---------------- FUTURE GRAPH ----------------
fig3 = plt.figure(figsize=(12,6))
plt.plot(future_predictions, 'g', label='Future')
plt.legend()
st.pyplot(fig3)
