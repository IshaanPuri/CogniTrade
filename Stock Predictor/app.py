from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from prophet import Prophet
import numpy as np
import pandas_ta as ta


app = Flask(__name__)

# Function to build the stock forecast model
def build_stock_forecast_model(stock_data):
    # Rename columns to match Prophet's requirement
    stock_data = stock_data.reset_index()
    stock_data = stock_data.rename(columns={'Date': 'ds', 'Close': 'y'})

    model = Prophet(daily_seasonality=True)
    model.fit(stock_data)
    return model

# Function to make stock price forecasts
def make_forecast(model, periods):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

# Function to format the forecast data for display
def format_forecast(forecast, num_periods):
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast = forecast.tail(num_periods)  # Only keep the forecast for the specified number of periods
    forecast['ds'] = forecast['ds'].dt.strftime('%Y-%m-%d')  # Convert the date to a more readable format
    forecast = forecast.rename(columns={'ds': 'Date', 'yhat': 'Forecasted Price', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'})
    
    # Add long-term prediction date
    forecast['Long-Term Prediction Date'] = forecast.iloc[-1]['Date']
    
    return forecast


# Function to analyze the forecast and provide buy/sell recommendations
def analyze_forecast(formatted_forecast, current_stock_price, short_term_threshold=7, long_term_days=365):
    analysis = ""
    
    # Short-term Analysis
    short_term_forecast = formatted_forecast.iloc[0]
    short_term_change = (short_term_forecast['Forecasted Price'] - current_stock_price) / current_stock_price * 100

    if abs(short_term_change) > short_term_threshold:
        analysis += f"In the short term, the forecasted stock price is {short_term_forecast['Forecasted Price']:.2f}, "
        if short_term_change > 0:
            analysis += f"indicating a potential +{short_term_change:.2f}% increase from the current price of {current_stock_price:.2f}. This suggests the stock may be a good buy for the short term.\n"
        else:
            analysis += f"indicating a potential {short_term_change:.2f}% decrease from the current price of {current_stock_price:.2f}. This suggests caution for short-term trading.\n"
    else:
        analysis += f"In the short term, the forecasted stock price is {short_term_forecast['Forecasted Price']:.2f}, indicating a relatively stable price around the current price of {current_stock_price:.2f}. No strong recommendation for short-term trading.\n"

    # Long-term Analysis
    long_term_data = formatted_forecast.head(long_term_days)
    long_term_trend = all(long_term_data['Forecasted Price'] > current_stock_price)

    if long_term_trend:
        analysis += f"In the long term, the forecast indicates a consistent upward trend over the next {long_term_days} days, suggesting the stock may be a good buy for long-term investment.\n"
    else:
        analysis += f"In the long term, the forecast does not show a consistent upward trend over the next {long_term_days} days. Caution is advised for long-term investment.\n"

    # Advanced Analysis - Moving Averages using pandas_ta
    historical_prices = formatted_forecast.iloc[:-long_term_days]['Forecasted Price']
    short_term_ma = ta.sma(historical_prices, length=10)
    long_term_ma = ta.sma(historical_prices, length=50)

    if short_term_ma is not None and long_term_ma is not None:
        if short_term_ma.iloc[-1] > long_term_ma.iloc[-1]:
            analysis += f"Short-term moving average (10-day) is above the long-term moving average (50-day), indicating a potential bullish trend.\n"
        else:
            analysis += f"Short-term moving average (10-day) is below the long-term moving average (50-day), indicating a potential bearish trend.\n"
    else:
        analysis += "Insufficient data to calculate moving averages for advanced analysis.\n"

    # Advanced Analysis - Relative Strength Index (RSI) using pandas_ta
    rsi = ta.rsi(historical_prices, length=14)
    if rsi is not None:
        if rsi.iloc[-1] > 70:
            analysis += f"RSI is currently {rsi.iloc[-1]:.2f}, suggesting the stock may be overbought. Caution is advised for buying at this level.\n"
        elif rsi.iloc[-1] < 30:
            analysis += f"RSI is currently {rsi.iloc[-1]:.2f}, suggesting the stock may be oversold. This could present a potential buying opportunity.\n"
        else:
            analysis += f"RSI is currently {rsi.iloc[-1]:.2f}, indicating a neutral sentiment in the market.\n"
    else:
        analysis += "Insufficient data to calculate RSI for advanced analysis.\n"

    return analysis

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the stock ticker symbol and forecast periods from the user input
        stock_ticker = request.form['stock_ticker']
        num_periods = int(request.form['num_periods'])  # Convert to integer

        # Fetch historical stock data using yfinance
        stock_data = yf.download(stock_ticker, period="1y")

        # Check if stock data is available for the given ticker symbol
        if stock_data.empty:
            return render_template("index.html", error="Invalid stock ticker symbol. Please try again.")

        # Build the stock forecast model
        stock_model = build_stock_forecast_model(stock_data)

        # Make stock price forecasts for the specified number of periods
        forecast_data = make_forecast(stock_model, periods=num_periods)

        # Format the forecast data for display on the webpage
        formatted_forecast = format_forecast(forecast_data, num_periods=num_periods)

        # Analyze the forecast and provide buy/sell recommendations
        current_stock_price = stock_data['Close'].iloc[-1]
        analysis = analyze_forecast(formatted_forecast, current_stock_price)

        # Add stock ticker symbol to the analysis
        analysis = f"Analysis for {stock_ticker.upper()}:\n\n{analysis}"

        return render_template("index.html", title="Stock Price Forecasting", stock_ticker=stock_ticker, forecast=formatted_forecast.to_html(index=False), analysis=analysis)

    return render_template("index.html", title="Stock Price Forecasting")

if __name__ == "__main__":
    app.run(debug=True)