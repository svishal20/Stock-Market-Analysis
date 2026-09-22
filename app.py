import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSage · ML Predictor",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.stApp { background: #0d1117; color: #e6edf3; }

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-label { font-size: 0.75rem; color: #8b949e; letter-spacing: 0.1em; text-transform: uppercase; }
.metric-value { font-size: 1.8rem; font-weight: 700; font-family: 'Space Mono', monospace; color: #58a6ff; }
.metric-sub   { font-size: 0.8rem; color: #8b949e; margin-top: 0.2rem; }

.tag-good  { color: #3fb950; font-weight: 600; }
.tag-mid   { color: #d29922; font-weight: 600; }
.tag-bad   { color: #f85149; font-weight: 600; }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #58a6ff;
    border-bottom: 1px solid #30363d;
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 StockSage")
    st.markdown("*ML-Powered Stock Analysis*")
    st.divider()

    ticker = st.text_input("Stock Ticker Symbol", value="AAPL",
                           help="e.g. AAPL, TSLA, RELIANCE.NS, TCS.NS, MSFT").upper().strip()

    period_map = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
    period_label = st.selectbox("Historical Data Period", list(period_map.keys()), index=1)
    period = period_map[period_label]

    predict_days = st.slider("Days to Predict Ahead", 5, 60, 30)

    st.divider()
    run = st.button("🚀 Run Prediction", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Data sourced from Yahoo Finance · For educational use only · Not financial advice")

# ── Helper functions ──────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

def style_axes(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

def fmt_dates(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 📈 StockSage · ML Stock Predictor")
st.markdown("Enter any stock ticker in the sidebar and hit **Run Prediction**.")

if not run:
    st.info("👈 Configure settings in the sidebar and click **Run Prediction** to begin.")
    st.stop()

# ── Fetch data ────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching data for **{ticker}**..."):
    try:
        stock = yf.Ticker(ticker)
        df    = stock.history(period=period)
        info  = stock.info
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

if df.empty:
    st.error(f"No data found for ticker **{ticker}**. Please check the symbol and try again.")
    st.stop()

df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
df.index = pd.to_datetime(df.index).tz_localize(None)

# ── Company header ────────────────────────────────────────────────────────────
company_name = info.get("longName", ticker)
st.markdown(f"## {company_name} ({ticker})")

col1, col2, col3, col4 = st.columns(4)
latest = df["Close"].iloc[-1]
prev   = df["Close"].iloc[-2]
change = latest - prev
pct    = (change / prev) * 100
color  = "tag-good" if change >= 0 else "tag-bad"
arrow  = "▲" if change >= 0 else "▼"

with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Current Price</div>
        <div class="metric-value">${latest:.2f}</div>
        <div class="metric-sub {color}">{arrow} {abs(change):.2f} ({abs(pct):.2f}%)</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">52-Week High</div>
        <div class="metric-value">${df['High'].max():.2f}</div>
        <div class="metric-sub">over {period_label.lower()}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">52-Week Low</div>
        <div class="metric-value">${df['Low'].min():.2f}</div>
        <div class="metric-sub">over {period_label.lower()}</div>
    </div>""", unsafe_allow_html=True)
with col4:
    avg_vol = df["Volume"].mean()
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Avg Daily Volume</div>
        <div class="metric-value">{avg_vol/1e6:.1f}M</div>
        <div class="metric-sub">shares/day</div>
    </div>""", unsafe_allow_html=True)

# ── Technical indicators ──────────────────────────────────────────────────────
df["MA_20"]  = df["Close"].rolling(20).mean()
df["MA_50"]  = df["Close"].rolling(50).mean()
df["RSI"]    = compute_rsi(df["Close"])
df["MACD"], df["Signal"], df["Histogram"] = compute_macd(df["Close"])
df["BB_mid"] = df["Close"].rolling(20).mean()
df["BB_std"] = df["Close"].rolling(20).std()
df["BB_up"]  = df["BB_mid"] + 2 * df["BB_std"]
df["BB_dn"]  = df["BB_mid"] - 2 * df["BB_std"]
df.dropna(inplace=True)

# ── Chart 1: Price + MAs + Bollinger Bands ────────────────────────────────────
st.markdown('<div class="section-title">📊 Price Chart with Technical Indicators</div>', unsafe_allow_html=True)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor="#0d1117",
                         gridspec_kw={"height_ratios": [3, 1, 1]})
plt.subplots_adjust(hspace=0.05)

# Price
ax1 = axes[0]
style_axes(ax1)
ax1.plot(df.index, df["Close"], color="#58a6ff", lw=1.5, label="Close Price")
ax1.plot(df.index, df["MA_20"], color="#3fb950", lw=1, linestyle="--", label="MA 20")
ax1.plot(df.index, df["MA_50"], color="#d29922", lw=1, linestyle="--", label="MA 50")
ax1.fill_between(df.index, df["BB_up"], df["BB_dn"], alpha=0.08, color="#58a6ff", label="Bollinger Bands")
ax1.plot(df.index, df["BB_up"], color="#58a6ff", lw=0.5, alpha=0.4)
ax1.plot(df.index, df["BB_dn"], color="#58a6ff", lw=0.5, alpha=0.4)
ax1.set_ylabel("Price (USD)", color="#8b949e")
ax1.legend(loc="upper left", fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
ax1.set_title(f"{ticker} · Price & Indicators", color="#e6edf3", fontsize=12, pad=10)
ax1.set_xticklabels([])

# RSI
ax2 = axes[1]
style_axes(ax2)
ax2.plot(df.index, df["RSI"], color="#bc8cff", lw=1)
ax2.axhline(70, color="#f85149", lw=0.8, linestyle="--", alpha=0.7)
ax2.axhline(30, color="#3fb950", lw=0.8, linestyle="--", alpha=0.7)
ax2.fill_between(df.index, df["RSI"], 70, where=(df["RSI"] >= 70), alpha=0.15, color="#f85149")
ax2.fill_between(df.index, df["RSI"], 30, where=(df["RSI"] <= 30), alpha=0.15, color="#3fb950")
ax2.set_ylim(0, 100)
ax2.set_ylabel("RSI", color="#8b949e")
ax2.set_xticklabels([])

# MACD
ax3 = axes[2]
style_axes(ax3)
colors_hist = ["#3fb950" if v >= 0 else "#f85149" for v in df["Histogram"]]
ax3.bar(df.index, df["Histogram"], color=colors_hist, alpha=0.6, width=1)
ax3.plot(df.index, df["MACD"],   color="#58a6ff", lw=1, label="MACD")
ax3.plot(df.index, df["Signal"], color="#f0883e", lw=1, label="Signal")
ax3.set_ylabel("MACD", color="#8b949e")
ax3.legend(loc="upper left", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
fmt_dates(ax3)

st.pyplot(fig)
plt.close()

# ── ML Model ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🤖 Linear Regression Model</div>', unsafe_allow_html=True)

# Features
feature_cols = ["Open", "High", "Low", "Volume", "MA_20", "MA_50", "RSI", "MACD", "Signal"]
df["Target"] = df["Close"].shift(-1)
df.dropna(inplace=True)

X = df[feature_cols].values
y = df["Target"].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100

def r2_tag(r):
    if r >= 0.9: return "tag-good", "Excellent"
    if r >= 0.7: return "tag-mid", "Good"
    return "tag-bad", "Needs Improvement"

tag_cls, tag_lbl = r2_tag(r2)

mc1, mc2, mc3, mc4 = st.columns(4)
for col, label, value, sub in [
    (mc1, "MAE",  f"${mae:.2f}",  "Mean Absolute Error"),
    (mc2, "RMSE", f"${rmse:.2f}", "Root Mean Sq Error"),
    (mc3, "MAPE", f"{mape:.1f}%", "Mean Abs % Error"),
    (mc4, "R² Score", f"{r2:.4f}", tag_lbl),
]:
    cls = tag_cls if label == "R² Score" else ""
    col.markdown(f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cls}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

# ── Chart 2: Actual vs Predicted ──────────────────────────────────────────────
st.markdown('<div class="section-title">🎯 Actual vs Predicted Prices</div>', unsafe_allow_html=True)

test_dates = df.index[-len(y_test):]
fig2, ax = plt.subplots(figsize=(14, 4), facecolor="#0d1117")
style_axes(ax)
ax.plot(test_dates, y_test, color="#58a6ff", lw=1.5, label="Actual Price")
ax.plot(test_dates, y_pred, color="#f0883e", lw=1.5, linestyle="--", label="Predicted Price")
ax.fill_between(test_dates, y_test, y_pred, alpha=0.1, color="#bc8cff")
ax.set_title("Test Set: Actual vs Predicted", color="#e6edf3", fontsize=11)
ax.set_ylabel("Price (USD)", color="#8b949e")
ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)
fmt_dates(ax)
st.pyplot(fig2)
plt.close()

# ── Future prediction ─────────────────────────────────────────────────────────
st.markdown(f'<div class="section-title">🔮 {predict_days}-Day Future Forecast</div>', unsafe_allow_html=True)

last_row   = df[feature_cols].iloc[-1].values.reshape(1, -1)
last_price = df["Close"].iloc[-1]
future_prices = []
current = last_row.copy()

for _ in range(predict_days):
    scaled   = scaler.transform(current)
    pred_val = model.predict(scaled)[0]
    future_prices.append(pred_val)
    # slide features forward simply
    current[0][0] = pred_val      # Open ≈ prev predicted
    current[0][1] = pred_val * 1.005
    current[0][2] = pred_val * 0.995
    # keep volume, indicators roughly the same

future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),
                             periods=predict_days, freq="B")
future_df = pd.DataFrame({"Predicted Close": future_prices}, index=future_dates)

# Plot
fig3, ax = plt.subplots(figsize=(14, 4), facecolor="#0d1117")
style_axes(ax)
history_tail = df["Close"].iloc[-60:]
ax.plot(history_tail.index, history_tail.values, color="#58a6ff", lw=1.5, label="Historical")
ax.plot(future_df.index, future_df["Predicted Close"], color="#3fb950", lw=2,
        linestyle="--", label=f"Forecast ({predict_days}d)", marker="o", markersize=3)
ax.axvline(df.index[-1], color="#8b949e", lw=0.8, linestyle=":")
ax.fill_between(future_df.index,
                future_df["Predicted Close"] * 0.97,
                future_df["Predicted Close"] * 1.03,
                alpha=0.15, color="#3fb950", label="±3% Confidence Band")
ax.set_title(f"{ticker} · {predict_days}-Day Forecast", color="#e6edf3", fontsize=11)
ax.set_ylabel("Price (USD)", color="#8b949e")
ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)
fmt_dates(ax)
st.pyplot(fig3)
plt.close()

# Table
end_price  = future_prices[-1]
total_chg  = end_price - last_price
total_pct  = (total_chg / last_price) * 100
direction  = "📈 Bullish" if total_chg >= 0 else "📉 Bearish"

col_a, col_b, col_c = st.columns(3)
col_a.markdown(f"""<div class="metric-card">
    <div class="metric-label">Forecast Start</div>
    <div class="metric-value">${last_price:.2f}</div>
    <div class="metric-sub">today's close</div>
</div>""", unsafe_allow_html=True)
col_b.markdown(f"""<div class="metric-card">
    <div class="metric-label">Forecast End ({predict_days}d)</div>
    <div class="metric-value">${end_price:.2f}</div>
    <div class="metric-sub">{'<span class="tag-good">' if total_chg>=0 else '<span class="tag-bad">'}
    {'+' if total_chg>=0 else ''}{total_pct:.2f}%</span></div>
</div>""", unsafe_allow_html=True)
col_c.markdown(f"""<div class="metric-card">
    <div class="metric-label">Signal</div>
    <div class="metric-value" style="font-size:1.4rem">{direction}</div>
    <div class="metric-sub">model-based signal</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📋 View Forecast Table"):
    future_df.index = future_df.index.strftime("%Y-%m-%d")
    future_df.index.name = "Date"
    future_df["Change from Today"] = future_df["Predicted Close"] - last_price
    future_df["Change %"] = ((future_df["Predicted Close"] - last_price) / last_price * 100).round(2)
    future_df["Predicted Close"] = future_df["Predicted Close"].round(2)
    future_df["Change from Today"] = future_df["Change from Today"].round(2)
    st.dataframe(future_df, use_container_width=True)

# ── Feature importance ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔬 Feature Importance (Model Coefficients)</div>', unsafe_allow_html=True)

coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": np.abs(model.coef_)
}).sort_values("Coefficient", ascending=True)

fig4, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
style_axes(ax)
bars = ax.barh(coef_df["Feature"], coef_df["Coefficient"],
               color=["#58a6ff" if c > coef_df["Coefficient"].median() else "#30363d"
                      for c in coef_df["Coefficient"]])
ax.set_xlabel("Absolute Coefficient", color="#8b949e")
ax.set_title("Feature Importance", color="#e6edf3", fontsize=11)
st.pyplot(fig4)
plt.close()

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.divider()
st.caption("⚠️ **Disclaimer:** This tool is built for educational/college project purposes only. "
           "Predictions are based on historical patterns and do not constitute financial advice. "
           "Never invest based solely on model predictions.")
