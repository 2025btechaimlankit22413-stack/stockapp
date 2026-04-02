HECK ----------------
if df is None or df.empty or 'Close' not in df.columns:
    st.error("❌ No valid data found. Check ticker.")
    st.stop()

# ---------------- CLEAN DATA ----------------
df = df[['Close']].copy()
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df.dropna(inplace=True)

