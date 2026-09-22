# 📈 StockSage — ML Stock Predictor
### College ML Project | Built with Python + Streamlit

---

## 🧠 What This Project Does

StockSage is a machine learning web application that predicts stock prices for **any publicly listed stock** using **Linear Regression** enriched with technical indicators.

---

## 🔧 Tech Stack

| Component       | Library/Tool          |
|-----------------|-----------------------|
| ML Model        | scikit-learn (Linear Regression) |
| Data Source     | yfinance (Yahoo Finance) |
| Web UI          | Streamlit             |
| Charts          | Matplotlib            |
| Data Processing | Pandas, NumPy         |

---

## 📊 Features

- **Any Stock** — Enter any ticker: `AAPL`, `TSLA`, `RELIANCE.NS`, `TCS.NS`, etc.
- **Technical Indicators**:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - 20-day & 50-day Moving Averages
- **ML Model Evaluation**:
  - MAE, RMSE, MAPE, R² Score
  - Actual vs Predicted chart on test data
- **Future Forecast** — Predict 5 to 60 days ahead
- **Feature Importance** — See which indicators influence predictions most

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Open in browser
Streamlit will open `http://localhost:8501` automatically.

---

## 🗂️ Project Structure

```
stock_predictor/
├── app.py             # Main Streamlit application
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## 📐 ML Methodology

### Features Used
| Feature   | Description                          |
|-----------|--------------------------------------|
| Open      | Opening price of the day             |
| High      | Highest price of the day             |
| Low       | Lowest price of the day              |
| Volume    | Number of shares traded              |
| MA_20     | 20-day simple moving average         |
| MA_50     | 50-day simple moving average         |
| RSI       | Momentum oscillator (0-100)          |
| MACD      | Trend-following momentum indicator   |
| Signal    | 9-day EMA of MACD                    |

### Target Variable
- **Next day's closing price** (`Close` shifted by 1 day)

### Train/Test Split
- **80% training**, 20% testing (time-series preserving — no shuffle)

### Evaluation Metrics
- **MAE** — Average dollar error
- **RMSE** — Penalizes large errors more
- **MAPE** — Percentage error (scale-independent)
- **R²** — How well the model fits (1.0 = perfect)

---

## 💡 Key Concepts (for viva/presentation)

1. **Why Linear Regression?** — Simple, interpretable, good baseline model
2. **Why MinMaxScaler?** — Features have different scales (price vs. volume); scaling helps the model learn equally
3. **Why no shuffle in train/test split?** — Stock data is time-series; shuffling would cause data leakage
4. **What are Bollinger Bands?** — Volatility bands placed 2 std deviations above/below the 20-day MA
5. **What is RSI?** — Values above 70 = overbought, below 30 = oversold

---

## ⚠️ Disclaimer
This project is for educational purposes only. Not financial advice.
