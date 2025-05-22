# Trading Agent System 📊

A comprehensive AI-powered trading assistant that combines fundamental analysis, technical analysis, macroeconomic insights, and news sentiment to provide intelligent investment recommendations and portfolio management.

## 🚀 Features

### Multi-Agent Analysis System
- **Market Data Agent**: Technical analysis, chart patterns, volume analysis, liquidity assessment
- **Fundamental Analysis Agent**: Financial statements, ratios, valuation models, earnings analysis
- **Macro Environment Agent**: Economic indicators, central bank analysis, sector rotation, geopolitical risks
- **News Sentiment Agent**: Financial news analysis, sentiment tracking, market impact assessment
- **Orchestrator Agent**: Coordinates all agents to provide comprehensive trading recommendations

### Portfolio Management
- Track your investment positions with detailed holdings
- Import/export portfolio data via CSV
- Visual portfolio allocation charts
- Automatic portfolio update suggestions based on agent recommendations
- Edit mode for manual position adjustments

### Streamlit Web Interface
- **Portfolio Tab**: Manage your holdings, view allocation, track changes
- **Analysis Tab**: Interactive queries with the AI trading agents
- **History Tab**: Review past analyses and recommendations

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Required API keys (optional, system works with simulated data):
  - News API
  - Alpha Vantage
  - Finnhub
  - FRED (Federal Reserve Economic Data)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TradingAgent.git
cd TradingAgent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory:
```bash
# API Keys (optional - system works without them using simulated data)
NEWS_API_KEY=your_news_api_key
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
FINNHUB_KEY=your_finnhub_key
FRED_API_KEY=your_fred_api_key
```

4. Install required NLTK data:
```python
import nltk
nltk.download('vader_lexicon')
```

## 🚀 Usage

### Web Interface (Recommended)
Launch the Streamlit app:
```bash
streamlit run app_w_portfolio.py
```

### Command Line Interface
Run the trading agent system directly:
```bash
python trading_agent_system.py
```

### Example Queries
- "What are the best growth stocks to buy right now?"
- "Analyze my current portfolio and suggest rebalancing"
- "What's the current market regime and how should I position?"
- "Find undervalued dividend stocks in the healthcare sector"
- "Should I sell any of my current holdings based on recent news?"

## 📋 File Structure

```
TradingAgent/
├── README.md
├── requirements.txt
├── .env (create this)
├── app_w_portfolio.py          # Streamlit web interface
├── trading_agent_system.py     # Main agent orchestration
├── market_data_tools.py        # Technical analysis tools
├── fundamental_analysis_tools.py # Company analysis tools
├── macro_environment_tools.py  # Economic analysis tools
├── news_analysis_tools.py      # News sentiment tools
└── trading_portfolio.csv       # Your portfolio data (auto-created)
```

## 🔧 Configuration

### API Keys
The system is designed to work with or without API keys:
- **With API keys**: Gets real-time data from financial APIs
- **Without API keys**: Uses simulated but realistic data for demonstration

### Supported APIs
- **News API**: Financial news and sentiment analysis
- **Alpha Vantage**: Stock prices, fundamentals, economic data
- **Finnhub**: Additional market data and news
- **FRED**: Federal Reserve economic data

## 📊 Agent Capabilities

### Market Data Agent
- Historical price analysis
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Support/resistance level identification
- Volume profile analysis
- Market structure assessment
- Liquidity conditions evaluation

### Fundamental Analysis Agent
- Financial statement analysis
- Key ratio calculations (P/E, ROE, debt ratios, etc.)
- Earnings report analysis
- DCF valuation models
- ESG metrics assessment
- Insider transaction analysis
- SEC filing parsing

### Macro Environment Agent
- Economic indicator tracking (GDP, inflation, unemployment)
- Central bank policy analysis
- Market regime identification
- Geopolitical risk assessment
- Sector rotation analysis
- Global liquidity conditions

### News Sentiment Agent
- Financial news search and analysis
- Sentiment scoring and categorization
- News impact on stock prices
- Economic news monitoring
- Market implications from news events

## 🎯 Portfolio Features

### Position Tracking
- Symbol, shares, entry price, date added
- Notes for each position
- Total cost and allocation calculations
- Visual pie charts for allocation

### Smart Updates
- Agent recommendations parsed for portfolio changes
- Buy/sell suggestions with approval workflow
- Automatic position updates based on recommendations
- Change history tracking

### Data Management
- CSV import/export functionality
- Local data storage
- Edit mode for manual adjustments
- Backup and restore capabilities

## 🔍 Example Workflows

### 1. Portfolio Analysis
```
Query: "Analyze my current portfolio and suggest improvements"
→ Macro Agent: Assesses current market regime
→ News Agent: Checks sentiment for your holdings
→ Fundamental Agent: Evaluates each position
→ Technical Agent: Reviews chart patterns
→ Result: Specific buy/sell/hold recommendations
```

### 2. Stock Discovery
```
Query: "Find promising AI stocks under $50"
→ Macro Agent: Identifies favorable sectors
→ News Agent: Finds stocks with positive sentiment
→ Fundamental Agent: Screens for strong financials
→ Technical Agent: Validates technical setups
→ Result: Ranked list of specific opportunities
```

## 🛡️ Risk Disclaimers

- This system is for educational and research purposes
- Not licensed financial advice
- Past performance doesn't guarantee future results
- Always do your own research before investing
- Consider consulting with a qualified financial advisor

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for improvement:
- Additional technical indicators
- More sophisticated valuation models
- Enhanced news sources
- Better portfolio optimization algorithms
- Mobile-responsive interface improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues:
1. Check that all dependencies are installed correctly
2. Verify your API keys are properly set (if using real data)
3. Review the console output for error messages
4. Create an issue on GitHub with details about the problem

## 🔮 Future Enhancements

- Real-time portfolio tracking with broker integration
- Advanced options and derivatives analysis
- Backtesting framework for strategy validation
- Mobile app development
- Integration with additional data providers
- Machine learning model improvements
- Risk management tools
- Tax optimization features

---

**Built with**: Python, Streamlit, yfinance, pandas, plotly, and various financial APIs

**Disclaimer**: This software is provided for educational purposes only and should not be considered as financial advice.
