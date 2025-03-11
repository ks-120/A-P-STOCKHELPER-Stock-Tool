import datetime
import tkinter as tk
from tkinter import ttk
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

now = datetime.datetime.now().strftime("%Y-%m-%d")

def calculate_financial_ratios():
    ticker_symbol = entry.get()
    ticker = yf.Ticker(ticker_symbol)

    financials = ticker.financials
    info = ticker.info

    eps_ratio = financials.loc["Net Income"] / info['sharesOutstanding']
    pe_ratio = info['trailingPE']
    debt_to_equity_ratio = info['debtToEquity']
    roe_ratio = info['returnOnEquity']
    quick_ratio = info['quickRatio']

    ratios = {
        "P/E Ratio": {
            "Definition": "The price-to-earnings (P/E) ratio is a valuation ratio that compares the market price per share to the earnings per share (EPS).",
            "Value": pe_ratio
        },
        "EPS Ratio": {
            "Definition": "The earnings per share (EPS) ratio measures the profitability of a company on a per-share basis.",
            "Value": eps_ratio
        },
        "Debt-to-Equity Ratio": {
            "Definition": "The debt-to-equity ratio is a financial ratio that compares a company's total debt to its shareholders' equity.",
            "Value": debt_to_equity_ratio
        },
        "ROE Ratio": {
            "Definition": "The return on equity (ROE) ratio measures a company's profitability by revealing how much profit a company generates with the money shareholders have invested.",
            "Value": roe_ratio
        },
        "Quick Ratio": {
            "Definition": "The quick ratio, also known as the acid-test ratio, is a liquidity ratio that measures a company's ability to cover its short-term liabilities with its most liquid assets.",
            "Value": quick_ratio
        }
    }

    # Create the main window
    root = tk.Tk()
    root.title("Financial Ratios")
    root.geometry("800x600")
    root.configure(bg="black")

    # Create a treeview to display the ratios
    tree = ttk.Treeview(root)
    tree["columns"] = ("Value", "Definition")
    tree.column("#0", width=200)
    tree.column("Value", width=200)
    tree.column("Definition", width=400)
    tree.heading("#0", text="Ratio")
    tree.heading("Value", text="Value")
    tree.heading("Definition", text="Definition")
    tree.pack(fill="both", expand=True)

    # Populate the treeview with the ratios
    for ratio_name, ratio_data in ratios.items():
        tree.insert("", "end", text=ratio_name, values=(ratio_data["Value"], ratio_data["Definition"]))

    root.mainloop()

def plot_stock_price():
    ticker_symbol = entry.get()
    ticker = yf.Ticker(ticker_symbol)

    historical_data = ticker.history(period="max", start="2002-01-01", end=now)

    # Create the main window
    root = tk.Tk()
    root.title("Historical Closing Price of " + ticker_symbol)
    root.geometry("800x600")
    root.configure(bg="black")

    # Create a canvas to plot the stock price
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(historical_data.index, historical_data["Close"])
    ax.set_title("Historical Closing Price of " + ticker_symbol)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)

    canvas = tk.Canvas(root)
    canvas.pack(fill="both", expand=True)

    # Add the plot to the canvas
    fig.savefig("stock_price_plot.png")
    plot_img = tk.PhotoImage(file="stock_price_plot.png")
    canvas.create_image(0, 0, anchor="nw", image=plot_img)

    root.mainloop()

def predict_stock_price():
    ticker_symbol = entry.get()
    ticker = yf.Ticker(ticker_symbol)

    data = yf.download(ticker_symbol, start="2002-01-01", end=now)

    close_prices = data["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)
    train_size = int(len(scaled_prices) * 0.8)
    train_data = scaled_prices[:train_size]
    test_data = scaled_prices[train_size:]

    def create_sequences(data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i : i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)

    sequence_length = 30
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    model_lstm = Sequential()
    model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
    model_lstm.add(LSTM(units=50))
    model_lstm.add(Dense(units=1))
    model_lstm.compile(optimizer="adam", loss="mean_squared_error")
    model_lstm.fit(X_train, y_train, epochs=20, batch_size=32)

    combined_data = np.concatenate((train_data, test_data))
    X_pred, _ = create_sequences(combined_data, sequence_length)
    X_pred = X_pred[-30:]
    predicted_prices_lstm = model_lstm.predict(X_pred)
    predicted_prices_lstm = scaler.inverse_transform(predicted_prices_lstm)

    X_train_svm = X_train.reshape((X_train.shape[0], -1))
    y_train_svm = y_train.reshape((y_train.shape[0],))
    svm_model = SVR(kernel='rbf')
    svm_model.fit(X_train_svm, y_train_svm)

    X_pred_svm = X_pred.reshape((X_pred.shape[0], -1))
    predicted_prices_svm = svm_model.predict(X_pred_svm)
    predicted_prices_svm = predicted_prices_svm.reshape((-1, 1))
    predicted_prices_svm = scaler.inverse_transform(predicted_prices_svm)

    X_train_rf = X_train.reshape((X_train.shape[0], -1))
    y_train_rf = y_train.reshape((y_train.shape[0],))
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train_rf, y_train_rf)

    X_pred_rf = X_pred.reshape((X_pred.shape[0], -1))
    predicted_prices_rf = rf_model.predict(X_pred_rf)
    predicted_prices_rf = predicted_prices_rf.reshape((-1, 1))
    predicted_prices_rf = scaler.inverse_transform(predicted_prices_rf)

    predicted_prices_combined = (predicted_prices_lstm.flatten() + predicted_prices_svm.flatten() + predicted_prices_rf.flatten()) / 3

    last_date = data.index[-1]
    date_range = pd.date_range(last_date, periods=30, freq="D")

    predictions_df_lstm = pd.DataFrame(predicted_prices_lstm.flatten(), columns=["LSTM Predicted Price"], index=date_range)
    predictions_df_svm = pd.DataFrame(predicted_prices_svm.flatten(), columns=["SVM Predicted Price"], index=date_range)
    predictions_df_rf = pd.DataFrame(predicted_prices_rf.flatten(), columns=["Random Forest Predicted Price"], index=date_range)
    predictions_df_combined = pd.DataFrame(predicted_prices_combined, columns=["Combined Predicted Price"], index=date_range)

    # Create the main window
    root = tk.Tk()
    root.title("Stock Price Prediction for " + ticker_symbol)
    root.geometry("1200x800")
    root.configure(bg="black")

    # Create a canvas to plot the predictions
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data["Close"], label="Actual Price")
    ax.plot(predictions_df_lstm.index, predictions_df_lstm["LSTM Predicted Price"], label="LSTM Predicted Price")
    ax.plot(predictions_df_svm.index, predictions_df_svm["SVM Predicted Price"], label="SVM Predicted Price")
    ax.plot(predictions_df_rf.index, predictions_df_rf["Random Forest Predicted Price"], label="Random Forest Predicted Price")
    ax.plot(predictions_df_combined.index, predictions_df_combined["Combined Predicted Price"], label="Combined Predicted Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Stock Price Prediction for " + ticker_symbol)
    ax.legend()

    canvas = tk.Canvas(root)
    canvas.pack(fill="both", expand=True)

    # Add the plot to the canvas
    fig.savefig("stock_price_prediction.png")
    plot_img = tk.PhotoImage(file="stock_price_prediction.png")
    canvas.create_image(0, 0, anchor="nw", image=plot_img)

    root.mainloop()

def perform_sentiment_analysis(stock_symbol):
    ticker = yf.Ticker(stock_symbol)

    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")

    data = ticker.history(start=start_date, end=end_date)

    price_changes = data['Close'].pct_change().dropna()
    volume_changes = data['Volume'].pct_change().dropna()

    sentiment = []
    for i in range(len(price_changes)):
        if price_changes[i] >= 0 and volume_changes[i] >= 0:
            sentiment.append(1.0)
        elif price_changes[i] < 0 and volume_changes[i] < 0:
            sentiment.append(-1.0)
        else:
            sentiment.append(0.0)

    x = price_changes.index
    y = sentiment

    # Create the main window
    root = tk.Tk()
    root.title("Sentiment Analysis for " + stock_symbol)
    root.geometry("800x600")
    root.configure(bg="black")

    # Create a canvas to plot the sentiment analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sentiment')
    ax.set_title('Sentiment Analysis for {}'.format(stock_symbol))
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks(np.arange(-1.0, 1.1, 0.2))
    ax.axhline(0, color='gray', linestyle='--')

    canvas = tk.Canvas(root)
    canvas.pack(fill="both", expand=True)

    # Add the plot to the canvas
    fig.savefig("sentiment_analysis.png")
    plot_img = tk.PhotoImage(file="sentiment_analysis.png")
    canvas.create_image(0, 0, anchor="nw", image=plot_img)

    root.mainloop()

def perform_risk_assessment(symbol):
    ticker_symbol = entry.get()
    ticker = yf.Ticker(ticker_symbol)
    stock_data = yf.download(ticker_symbol, start='2020-01-01', end=now)

    stock_data['Log Returns'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    volatility = stock_data['Log Returns'].std() * np.sqrt(252)

    liquidity = stock_data['Volume'].mean()

    std_deviation = stock_data['Close'].std()

    # Create the main window
    root = tk.Tk()
    root.title("Risk Assessment for " + ticker_symbol)
    root.geometry("800x600")
    root.configure(bg="black")

    # Create labels to display the calculated factors and risk assessment result
    volatility_label = tk.Label(root, text="Volatility: {:.4f}".format(volatility), fg="white", bg="black", font=("Arial", 14))
    volatility_label.pack(pady=10)

    liquidity_label = tk.Label(root, text="Liquidity: {:.2f}".format(liquidity), fg="white", bg="black", font=("Arial", 14))
    liquidity_label.pack(pady=10)

    std_deviation_label = tk.Label(root, text="Standard Deviation: {:.2f}".format(std_deviation), fg="white", bg="black", font=("Arial", 14))
    std_deviation_label.pack(pady=10)

    if volatility > 0.3 and liquidity > 50000000 and std_deviation > 10:
        risk_assessment = "High Risk"
    elif volatility > 0.2 and liquidity > 10000000 and std_deviation > 5:
        risk_assessment = "Medium Risk"
    else:
        risk_assessment = "Low Risk"

    risk_label = tk.Label(root, text="Risk Assessment: " + risk_assessment, fg="white", bg="black", font=("Arial", 20, "bold"))
    risk_label.pack(pady=20)

    root.mainloop()

def open_catalog_page():
    catalog_window = tk.Toplevel(window)
    catalog_window.title("Catalog Page")
    catalog_window.configure(background="black")

    # Create buttons for each function
    button1 = tk.Button(catalog_window, text="Plot Stock Price", command=plot_stock_price, fg="white", bg="gray", font=("Arial", 14))
    button1.pack(pady=10)

    button2 = tk.Button(catalog_window, text="Predict Stock Price", command=predict_stock_price, fg="white", bg="gray", font=("Arial", 14))
    button2.pack(pady=10)

    button3 = tk.Button(catalog_window, text="Calculate Financial Ratios", command=calculate_financial_ratios, fg="white", bg="gray", font=("Arial", 14))
    button3.pack(pady=10)

    button4 = tk.Button(catalog_window, text="Perform Sentiment Analysis", command=lambda: perform_sentiment_analysis(entry.get()), fg="white", bg="gray", font=("Arial", 14))
    button4.pack(pady=10)

    button5 = tk.Button(catalog_window, text="Perform Risk Assessment", command=lambda: perform_risk_assessment(entry.get()), fg="white", bg="gray", font=("Arial", 14))
    button5.pack(pady=10)

# Create the main window
window = tk.Tk()
window.title("A&P STOCKHELPER")
window.geometry("800x600")
window.configure(background="black")

# Create a label for the project name
# Create an entry field for stock symbol input
entry = tk.Entry(window, width=10)
entry.pack(pady=10)
