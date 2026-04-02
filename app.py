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

if df is None or df.empty or 'Close' not in df.columns:
    st.error("❌ Data load failed")
    st.stop()

df = df[['Close']].copy()
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df.dropna(inplace=True)

if len(df) < 10:
    st.error("❌ Not enough data")
    st.stop()

# ---------------- CHART ----------------
st.subheader('Closing Price')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
st.pyplot(fig)

# ---------------- MODEL ----------------
df['Prediction'] = df['Close'].shift(-1)
df.dropna(inplace=True)

X = df[['Close']].values.astype(float)
y = df['Prediction'].values.astype(float)

model = LinearRegression()
model.fit(X, y)

# ---------------- CURRENT PRICE ----------------
current_price = float(df['Close'].iloc[-1])

st.success(f"Current Price: ${current_price:.2f}")

# ---------------- SIDEBAR ----------------
future_days = st.slider("Future Days", 5, 30, 10)

# ---------------- FUTURE PREDICTION (SAFE LOOP) ----------------
st.subheader("🔮 Future Prediction")

future_predictions = []
current_input = float(current_price)

for i in range(future_days):

    try:
        # ensure scalar float
        value = float(current_input)

        # check valid number
        if not np.isfinite(value):
            break

        # correct shape
        input_array = np.array([[value]], dtype=np.float64)

        # predict
        pred = model.predict(input_array)

        pred_value = float(pred[0])

        # check valid output
        if not np.isfinite(pred_value):
            break

        future_predictions.append(pred_value)

        # update input
        current_input = pred_value

    except Exception as e:
        st.error(f"Stopped at step {i}: {e}")
        break

# ---------------- OUTPUT ----------------
if len(future_predictions) == 0:
    st.warning("⚠️ Prediction failed")
else:
    st.write(future_predictions)

    # graph
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(future_predictions, label='Future')
    plt.legend()
    st.pyplot(fig2)

    # dataframe
    from datetime import datetime
    dates = pd.bdate_range(start=datetime.now(), periods=len(future_predictions))

    future_df = pd.DataFrame({
        "Date": dates,
        "Predicted Price": future_predictions
    })

    st.write(future_df)
