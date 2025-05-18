import os
import logging
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv

# Set up very basic console logging
print("Script starting...")

# Try to load environment variables
try:
    print("Loading .env file...")
    load_dotenv()
    print("Environment variables loaded.")
except Exception as e:
    print(f"Error loading .env file: {str(e)}")

# Print the API keys (masked for security)
fred_key = os.getenv('FRED_API_KEY', '')
alpha_key = os.getenv('ALPHA_VANTAGE_KEY', '')
news_key = os.getenv('NEWS_API_KEY', '')

print(f"FRED API Key present: {'Yes' if fred_key else 'No'}")
print(f"Alpha Vantage API Key present: {'Yes' if alpha_key else 'No'}")
print(f"News API Key present: {'Yes' if news_key else 'No'}")

# Test FRED API
print("\n--- Testing FRED API ---")
if fred_key:
    try:
        print("Making request to FRED API...")
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "UNRATE",  # Unemployment Rate
            "api_key": fred_key,
            "file_type": "json",
            "observation_start": "2023-01-01",
            "observation_end": datetime.now().strftime('%Y-%m-%d'),
        }
        
        response = requests.get(url, params=params)
        print(f"FRED API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            observations = data.get('observations', [])
            print(f"Retrieved {len(observations)} observations from FRED")
            if observations:
                print(f"Latest unemployment data: {observations[-1]['date']} - {observations[-1]['value']}%")
        else:
            print(f"FRED API Error: {response.text[:200]}")
    except Exception as e:
        print(f"Error with FRED API: {str(e)}")
else:
    print("Skipping FRED API test - no API key provided")

# Test Yahoo Finance API (for market data)
print("\n--- Testing Yahoo Finance API ---")
try:
    print("Fetching S&P 500 data from Yahoo Finance...")
    spy = yf.Ticker("SPY")
    history = spy.history(period="1mo")
    
    if not history.empty:
        print(f"Retrieved {len(history)} days of S&P 500 data")
        latest_close = history['Close'].iloc[-1]
        print(f"Latest S&P 500 close: {latest_close:.2f}")
    else:
        print("No data retrieved from Yahoo Finance")
except Exception as e:
    print(f"Error with Yahoo Finance API: {str(e)}")

# Test Alpha Vantage API
print("\n--- Testing Alpha Vantage API ---")
if alpha_key:
    try:
        print("Making request to Alpha Vantage API...")
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={alpha_key}"
        
        response = requests.get(url)
        print(f"Alpha Vantage API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if "Global Quote" in data:
                quote = data["Global Quote"]
                print(f"Apple stock price: {quote.get('05. price', 'N/A')}")
            else:
                print(f"Alpha Vantage API response: {data}")
        else:
            print(f"Alpha Vantage API Error: {response.text[:200]}")
    except Exception as e:
        print(f"Error with Alpha Vantage API: {str(e)}")
else:
    print("Skipping Alpha Vantage API test - no API key provided")

print("\n--- Testing News API ---")
if news_key:
    try:
        print("Making request to News API...")
        url = f"https://newsapi.org/v2/top-headlines?country=us&category=business&apiKey={news_key}"
        
        response = requests.get(url)
        print(f"News API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            print(f"Retrieved {len(articles)} news articles")
            if articles:
                print(f"Latest headline: {articles[0]['title']}")
        else:
            print(f"News API Error: {response.text[:200]}")
    except Exception as e:
        print(f"Error with News API: {str(e)}")
else:
    print("Skipping News API test - no API key provided")
    
print("\nScript completed.")