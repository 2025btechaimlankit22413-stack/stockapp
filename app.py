import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG (MUST BE FIRST) ----------------
st.set_page_config(layout="wide")

# ---------------- TITLE ----------------
st.title('📈 Stock Trend Prediction App')

# ---------------- INPUT ----------------
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

start = '2010-01-01'
end = '2026-03-26'

# ---------------- DATA FETCH ----------------
@st.cache_data
def load_data(ticker):
    return yf.download(ticker, start=start, end=end, progress=False)

df = load_data(user_input)

# ---------------- CHECK DATA ----------------
if df is None or df.empty:
    st.error("❌ No data found. Please check the ticker symbol.")
    st.stop()

# ---------------- DATA DISPLAY ----------------
st.subheader('📊 Data Summary')
st.write(df.describe())

# ---------------- CHARTS ----------------
st.subheader('📉 Closing Price')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.title("Closing Price")
st.pyplot(fig)

# ---------------- MOVING AVERAGES ----------------
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

st.subheader('📊 Price with 100 MA')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Close')
plt.plot(ma100, label='MA100')
plt.legend()
st.pyplot(fig)

st.subheader('📊 Price with 100 MA & 200 MA')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Close')
plt.plot(ma100, label='MA100')
plt.plot(ma200, label='MA200')
plt.legend()
st.pyplot(fig)

# ---------------- PREPARE DATA ----------------
df_model = df[['Close']].copy()
df_model['Prediction'] = df_model['Close'].shift(-1)

df_model.dropna(inplace=True)

X = np.array(df_model[['Close']])
y = np.array(df_model['Prediction'])

# ---------------- TRAIN MODEL ----------------
model = LinearRegression()
model.fit(X, y)

# ---------------- PREDICTION ----------------
y_pred = model.predict(X)

# ---------------- CURRENT PRICE ----------------
try:
    current_price = float(df['Close'].dropna().iloc[-1])
except:
    current_price = 0
    st.warning("⚠️ Could not fetch current price.")

# ---------------- GRAPH ----------------
st.subheader('📈 Prediction vs Actual')

fig2 = plt.figure(figsize=(12,6))
plt.plot(y, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.axhline(y=current_price, linestyle='--', label='Current Price')
plt.legend()
plt.grid(alpha=0.3)
st.pyplot(fig2)

# ---------------- METRICS ----------------
if len(df) > 1:
    previous_price = float(df['Close'].iloc[-2])
    change = current_price - previous_price
    percent = (change / previous_price) * 100
else:
    change = 0
    percent = 0

col1, col2 = st.columns(2)

with col1:
    st.metric("💰 Price", f"${current_price:.2f}")

with col2:
    st.metric("📊 Change", f"{change:.2f}", f"{percent:.2f}%")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")
future_days = st.sidebar.slider("Future Days", 5, 60, 10)

# ---------------- FUTURE PREDICTION ----------------
st.subheader('🔮 Future Predictions')

last_price = df['Close'].iloc[-1]
future_predictions = []

current_input = last_price

for i in range(future_days):
    pred = model.predict([[current_input]])[0]
    future_predictions.append(pred)
    current_input = pred

# ---------------- FUTURE DATAFRAME ----------------
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
plt.plot(future_predictions, label='Future Prediction')
plt.legend()
plt.grid(alpha=0.3)
st.pyplot(fig3)
