import os
import requests
import yfinance as yf
from dotenv import load_dotenv
import fundamental_tools

print("Starting Fundamental API Test Script...")

# Load environment variables
load_dotenv()

# Get API keys
fmp_key = os.getenv('FMP_API_KEY', '')
alpha_key = os.getenv('ALPHA_VANTAGE_KEY', '')

print(f"FMP API Key present: {'Yes' if fmp_key else 'No'}")
print(f"Alpha Vantage API Key present: {'Yes' if alpha_key else 'No'}")

# Test ticker
test_ticker = "AAPL"
print(f"\nUsing test ticker: {test_ticker}")

# Test FMP API
print("\n--- Testing Financial Modeling Prep API ---")
if fmp_key:
    try:
        # Test income statement
        income_url = f"https://financialmodelingprep.com/api/v3/income-statement/{test_ticker}?apikey={fmp_key}&limit=1"
        response = requests.get(income_url)
        print(f"FMP Income Statement API Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data:
                print(f"Revenue: ${data[0]['revenue']:,}")
                print(f"Net Income: ${data[0]['netIncome']:,}")
                print(f"EPS: ${data[0]['eps']}")
            else:
                print("No income statement data returned")
        
        # Test financial ratios
        ratios_url = f"https://financialmodelingprep.com/api/v3/ratios/{test_ticker}?apikey={fmp_key}&limit=1"
        response = requests.get(ratios_url)
        print(f"FMP Ratios API Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data:
                print(f"P/E Ratio: {data[0].get('priceEarningsRatio', 'N/A')}")
                print(f"ROE: {data[0].get('returnOnEquity', 'N/A')}")
                print(f"Current Ratio: {data[0].get('currentRatio', 'N/A')}")
            else:
                print("No ratios data returned")
    except Exception as e:
        print(f"Error with FMP API: {str(e)}")
else:
    print("Skipping FMP API test - no API key provided")

# Test Alpha Vantage API
print("\n--- Testing Alpha Vantage Fundamental Data API ---")
if alpha_key:
    try:
        # Company Overview
        overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={test_ticker}&apikey={alpha_key}"
        response = requests.get(overview_url)
        print(f"Alpha Vantage Overview API Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data and 'Symbol' in data:
                print(f"Company: {data.get('Name')}")
                print(f"P/E Ratio: {data.get('PERatio')}")
                print(f"Market Cap: ${data.get('MarketCapitalization'):,}")
                print(f"EPS: ${data.get('EPS')}")
            else:
                print(f"Alpha Vantage API response: {data}")
                if 'Note' in data:
                    print("API limit reached or other issue. See note above.")
        
        # Income Statement (if not rate limited)
        if 'Note' not in data:
            income_url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={test_ticker}&apikey={alpha_key}"
            response = requests.get(income_url)
            print(f"Alpha Vantage Income Statement API Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data and 'annualReports' in data and data['annualReports']:
                    report = data['annualReports'][0]
                    print(f"Total Revenue: ${report.get('totalRevenue', 'N/A')}")
                    print(f"Net Income: ${report.get('netIncome', 'N/A')}")
                else:
                    print("No income statement data or API limit reached")
    except Exception as e:
        print(f"Error with Alpha Vantage API: {str(e)}")
else:
    print("Skipping Alpha Vantage API test - no API key provided")

# Test Yahoo Finance
print("\n--- Testing Yahoo Finance Fundamental Data ---")
try:
    ticker = yf.Ticker(test_ticker)
    
    # Get financial data
    info = ticker.info
    
    print(f"Yahoo Finance Test - {test_ticker} Fundamentals:")
    print(f"Company: {info.get('shortName', 'N/A')}")
    print(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
    print(f"Forward P/E: {info.get('forwardPE', 'N/A')}")
    print(f"Market Cap: ${info.get('marketCap', 0):,.0f}")
    
    # Test financial statements
    financials = ticker.financials
    if not financials.empty:
        print("\nFinancial Statement Data Available:")
        print(f"Number of periods: {financials.shape[1]}")
        if 'Total Revenue' in financials.index:
            print(f"Latest Revenue: ${financials.loc['Total Revenue'].iloc[0]:,.0f}")
    else:
        print("No financial statement data available")
    
    # Test balance sheet
    balance_sheet = ticker.balance_sheet
    if not balance_sheet.empty:
        print("\nBalance Sheet Data Available:")
        print(f"Number of periods: {balance_sheet.shape[1]}")
        if 'Total Assets' in balance_sheet.index:
            print(f"Total Assets: ${balance_sheet.loc['Total Assets'].iloc[0]:,.0f}")
    else:
        print("No balance sheet data available")
    
except Exception as e:
    print(f"Error with Yahoo Finance: {str(e)}")

print("\nFundamental API Test Script Completed.")