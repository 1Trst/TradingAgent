# TradingGPT - AI Trading Assistant

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/yourusername/TradingGPT/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

TradingGPT is an advanced AI-powered trading assistant that leverages multiple specialized agents to provide comprehensive market analysis, investment recommendations, and portfolio management features.

## 🚀 Overview

TradingGPT combines technical analysis, fundamental analysis, macroeconomic insights, and news sentiment to deliver well-rounded trading recommendations. The system uses a multi-agent architecture where specialized agents collaborate to analyze different aspects of the market and provide holistic insights.

Key features include:
- Technical analysis with multiple indicators and market structure assessment
- Fundamental analysis including financial ratios and valuation models
- Macroeconomic environment monitoring and central bank policy analysis 
- News sentiment analysis across various financial topics
- Portfolio tracking and management
- Interactive web interface built with Streamlit

## 🤖 Agent Architecture

TradingGPT uses a sophisticated multi-agent system:

1. **Trading Orchestrator** - Coordinates between specialized agents and synthesizes recommendations
2. **Market Data Analyst** - Performs technical analysis, chart patterns, and market structure assessment
3. **Fundamental Analyst** - Analyzes company financials, valuations, and business health
4. **Macro Environment Analyst** - Monitors economic indicators, central bank policies, and geopolitical risks
5. **News Sentiment Analyst** - Tracks sentiment in financial news and its impact on markets

## 📊 Technical Features

### Market Data Tools
- Historical price data analysis
- Multiple technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- Support/resistance identification
- Volume profile analysis
- Market structure assessment
- Liquidity analysis

### Fundamental Analysis Tools
- Financial statement analysis
- Financial ratio calculations
- Earnings report analysis
- Valuation models (DCF, etc.)
- Industry and competitive positioning assessment

### Macro Environment Tools
- Economic indicator tracking
- Central bank policy analysis
- Market regime assessment
- Geopolitical risk monitoring
- Sector rotation analysis

### News Analysis Tools
- Financial news search and sentiment analysis
- Company-specific news impact assessment
- Economic news monitoring
- Social media sentiment tracking (planned)

## 💻 Installation

### Prerequisites

- Python 3.8+
- Required API Keys:
  - News API (optional)
  - Alpha Vantage (optional)
  - FRED API (optional)

### Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/TradingGPT.git
cd TradingGPT
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up your API keys
```bash
# Create a .env file
cp .env.example .env
# Edit the .env file with your API keys
```

## 🏃‍♀️ Running the Application

### Streamlit Web Interface

```bash
streamlit run app_w_portfolio.py
```

This will launch the TradingGPT web interface where you can:
- Ask questions about stocks, sectors, or market conditions
- Track and manage your investment portfolio
- Analyze specific securities using all agents

### Command Line Interface

```bash
python trading_agent_system.py
```

This will start the command-line version of TradingGPT, allowing you to interact with the system directly.

## 📝 Example Queries

TradingGPT can answer questions like:

- "What stocks are positioned to benefit from current interest rate trends?"
- "Analyze the technical indicators for AAPL and give me a trading recommendation"
- "What sectors look promising given the current macroeconomic environment?"
- "How is recent news affecting Tesla's stock price?"
- "Rebalance my portfolio for the current market conditions"
- "What are the key support and resistance levels for MSFT?"

## 🧩 Architecture

The system is built using:
- **OpenAI Agent SDK** for the multi-agent architecture
- **yfinance** for market data
- **NLTK** for sentiment analysis
- **pandas** and **numpy** for data processing
- **Streamlit** for the web interface
- **Plotly** for interactive visualizations

## 🌱 Future Enhancements

- Risk management agent for portfolio optimization
- Options analysis capabilities
- Backtesting framework for strategy validation
- Enhanced social media sentiment analysis
- Real-time alerts for price movements and news

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

TradingGPT is for informational purposes only and does not constitute financial advice. Always do your own research and consider consulting with a financial advisor before making investment decisions.
