import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from sklearn.linear_model import LinearRegression

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide")

st.title('📈 Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# ---------------- DATA ----------------
df = yf.download(user_input, start='2010-01-01', end='2026-03-26', progress=False)

# HARD SAFETY CHECK
if df is None or df.empty or 'Close' not in df.columns:
    st.error("❌ Data load failed. Try another ticker.")
    st.stop()

df = df[['Close']].copy()
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna()

if len(df) < 10:
    st.error("❌ Not enough valid data.")
    st.stop()

# ---------------- CHART ----------------
st.subheader('Closing Price')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
st.pyplot(fig)

# ---------------- MODEL ----------------
df['Prediction'] = df['Close'].shift(-1)
df = df.dropna()

X = df[['Close']].values.astype(float)
y = df['Prediction'].values.astype(float)

# FINAL SAFETY (important)
mask = np.isfinite(X.flatten()) & np.isfinite(y)
X = X[mask]
y = y[mask]

if len(X) < 5:
    st.error("❌ Clean data too small.")
    st.stop()

model = LinearRegression()
model.fit(X, y)

# ---------------- CURRENT PRICE ----------------
current_price = float(df['Close'].iloc[-1])

st.success(f"Current Price: ${current_price:.2f}")

# ---------------- FUTURE PREDICTION ----------------
st.subheader("🔮 Future Prediction")

future_days = st.slider("Days", 5, 30, 10)

future_predictions = []
current_input = float(current_price)

for i in range(future_days):

    # FULL SAFETY BLOCK
    if not np.isfinite(current_input):
        break

    try:
        input_array = np.array([[current_input]], dtype=np.float64)

        # FINAL CHECK
        if not np.isfinite(input_array).all():
            break

        pred = model.predict(input_array)

        if not np.isfinite(pred).all():
            break

        pred_value = float(pred[0])

        future_predictions.append(pred_value)
        current_input = pred_value

    except Exception as e:
        st.error(f"Prediction stopped: {e}")
        break

# ---------------- OUTPUT ----------------
if len(future_predictions) == 0:
    st.warning("⚠️ No predictions generated.")
else:
    st.write(future_predictions)

    fig2 = plt.figure(figsize=(12,6))
    plt.plot(future_predictions)
    plt.title("Future Prediction")
    st.pyplot(fig2)
