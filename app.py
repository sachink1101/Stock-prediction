from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)  # Allow all origins

# Load trained models
scaler = joblib.load("./model/scaler.pkl")
xgb_model = joblib.load("./model/xgb_model.pkl")
lstm_model = load_model("./model/lstm_model.h5")

# Fetch Stock Data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y", interval="1d")
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    
    if df.empty:
        return None
    
    df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['close'], window=200)
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['MACD'] = ta.trend.macd(df['close'])
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    
    df.dropna(inplace=True)
    return df

# Predict stock price
def predict_price(df):
    features = ['close', 'SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'VWAP', 'ATR']
    
    if df is None or len(df) < 60:
        return None

    recent_data = df[features].values[-60:]
    recent_scaled = scaler.transform(recent_data)
    
    lstm_pred = lstm_model.predict(np.array([recent_scaled]))[0][0]
    xgb_pred = xgb_model.predict(recent_scaled.reshape(1, -1))[0]
    
    final_pred = (lstm_pred * 0.6 + xgb_pred * 0.4)
    final_pred = scaler.inverse_transform(np.array([[final_pred] * len(features)]))[0][0]
    
    return final_pred

# Trading Strategy
def trade_action(price, prev_close, rsi, macd, sma50, sma200, atr):
    if price > sma50 > sma200 and rsi < 30 and macd > 0:
        return "🔹 STRONG BUY"
    elif price > sma50 and rsi < 50 and macd > 0:
        return "🟢 BUY"
    elif price < sma50 < sma200 and rsi > 70 and macd < 0:
        return "🔻 STRONG SELL"
    elif price < sma50 and rsi > 55 and macd < 0:
        return "🔴 SELL"
    elif atr > prev_close * 0.02:
        return "⚠️ WAIT - High Volatility"
    else:
        return "⏳ HOLD"

# API Endpoint for multiple stocks
@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Stock Prediction API. Use POST /predict with stock tickers."})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    tickers = data.get("tickers", [])

    if not tickers or not isinstance(tickers, list):
        return jsonify({"error": "Invalid request. Provide a list of stock tickers."}), 400

    results = []

    for ticker in tickers:
        df = get_stock_data(ticker)
        if df is None:
            results.append({"ticker": ticker, "error": "Stock data not found."})
            continue

        predicted_price = predict_price(df)
        if predicted_price is None:
            results.append({"ticker": ticker, "error": "Not enough data to make a prediction."})
            continue

        current_price = df['close'].iloc[-1]
        support = df['low'].rolling(window=50).min().iloc[-1]
        resistance = df['high'].rolling(window=50).max().iloc[-1]
        buy_price = current_price * 1.05
        sell_price = current_price * 1.10
        stop_loss = current_price * 0.95
        target_price = current_price * 1.15

        action = trade_action(
            current_price, df['close'].iloc[-2], df['RSI'].iloc[-1], 
            df['MACD'].iloc[-1], df['SMA_50'].iloc[-1], df['SMA_200'].iloc[-1], df['ATR'].iloc[-1]
        )

        results.append({
            "ticker": ticker,
            "current_price": float(round(current_price, 2)),
            "predicted_price": float(round(predicted_price, 2)),
            "support": float(round(support, 2)),
            "resistance": float(round(resistance, 2)),
            "suggested_action": action,
            "buy_price": float(round(buy_price, 2)),
            "sell_price": float(round(sell_price, 2)),
            "stop_loss": float(round(stop_loss, 2)),
            "target_price": float(round(target_price, 2)),
        })

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
