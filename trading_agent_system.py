import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, RunConfig, WebSearchTool

# Import our tool implementations
from market_data_tools import MarketDataTools
# Assuming these are implemented
from fundamental_analysis_tools import FundamentalAnalysisTools
from macro_environment_tools import MacroEnvironmentTools

# Import the News Analysis Tools
from news_analysis_tools import NewsAnalysisTools

# Load environment variables
load_dotenv()

# Initialize tool instances
market_tools = MarketDataTools()
fundamental_tools = FundamentalAnalysisTools()
macro_tools = MacroEnvironmentTools()

# Function tools for the Market Data Agent
@function_tool
def get_historical_prices(symbol: str) -> str:
    """
    Fetch historical price data for a given symbol
    
    Args:
        symbol: The ticker symbol to fetch data for
    """
    result = market_tools.get_historical_prices(symbol, interval="1d", period="1mo")
    return str(result)

@function_tool
def get_price_data_with_timeframe(symbol: str, interval: str, period: str) -> str:
    """
    Fetch historical price data for a given symbol with custom timeframe
    
    Args:
        symbol: The ticker symbol to fetch data for
        interval: Time interval between data points (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
        period: The time period to fetch data for (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)
    """
    result = market_tools.get_historical_prices(symbol, interval, period)
    return str(result)

@function_tool
def calculate_technical_indicators(symbol: str) -> str:
    """
    Calculate common technical indicators for a given symbol
    
    Args:
        symbol: The ticker symbol to analyze
    """
    result = market_tools.calculate_technical_indicators(symbol, indicators=["sma", "rsi", "macd", "bollinger"], 
                                                       interval="1d", period="6mo")
    return str(result)

@function_tool
def calculate_specific_indicators(symbol: str, indicators: str, interval: str, period: str) -> str:
    """
    Calculate specific technical indicators for a given symbol
    
    Args:
        symbol: The ticker symbol to analyze
        indicators: Comma-separated list of indicators to calculate (e.g., "sma,rsi,macd")
        interval: Time interval between data points
        period: The time period to analyze
    """
    indicator_list = indicators.split(",") if indicators else None
    result = market_tools.calculate_technical_indicators(symbol, indicator_list, interval, period)
    return str(result)

@function_tool
def analyze_market_structure(symbol: str) -> str:
    """
    Analyze market structure including support/resistance, trends, and patterns
    
    Args:
        symbol: The ticker symbol to analyze
    """
    result = market_tools.analyze_market_structure(symbol, interval="1d", period="1y")
    return str(result)

@function_tool
def analyze_volume_profile(symbol: str) -> str:
    """
    Analyze volume profile and volume-based indicators
    
    Args:
        symbol: The ticker symbol to analyze
    """
    result = market_tools.analyze_volume_profile(symbol, interval="1d", period="3mo")
    return str(result)

@function_tool
def assess_liquidity_conditions(symbol: str) -> str:
    """
    Assess market liquidity conditions including volume analysis and trading costs
    
    Args:
        symbol: The ticker symbol to analyze
    """
    result = market_tools.assess_liquidity_conditions(symbol, interval="1d", period="3mo")
    return str(result)

# Function tools for the Fundamental Analysis Agent
@function_tool
def get_financial_statements(ticker: str) -> str:
    """
    Retrieve company financial statements (income statement, balance sheet, cash flow)
    
    Args:
        ticker: Company ticker symbol
    """
    result = fundamental_tools.get_financial_statements(ticker, statement_type="all", period="annual", years=3)
    return str(result)

@function_tool
def calculate_financial_ratios(ticker: str) -> str:
    """
    Calculate key financial ratios based on company data
    
    Args:
        ticker: Company ticker symbol
    """
    result = fundamental_tools.calculate_financial_ratios(ticker, ratio_categories=["all"], benchmark_against=None)
    return str(result)

@function_tool
def analyze_earnings_reports(ticker: str) -> str:
    """
    Extract and analyze data from recent earnings reports and calls
    
    Args:
        ticker: Company ticker symbol
    """
    result = fundamental_tools.analyze_earnings_reports(ticker, quarters_back=4, include_call_transcripts=False)
    return str(result)

@function_tool
def run_valuation_model(ticker: str) -> str:
    """
    Run financial valuation models on a company
    
    Args:
        ticker: Company ticker symbol
    """
    result = fundamental_tools.run_valuation_model(ticker, models=["dcf"], forecast_years=5)
    return str(result)

# Function tools for the Macro Environment Agent
@function_tool
def get_economic_indicators_us() -> str:
    """
    Retrieve key economic indicators for the US
    """
    indicators = ["gdp", "inflation", "unemployment", "interest_rates"]
    result = macro_tools.get_economic_indicators(indicators, country="US", period="5y")
    return str(result)

@function_tool
def get_central_bank_analysis_fed() -> str:
    """
    Analyze Federal Reserve communications and policy trends
    """
    result = macro_tools.get_central_bank_analysis(central_bank="federal_reserve", lookback_months=6)
    return str(result)

@function_tool
def get_market_regime_analysis_current() -> str:
    """
    Analyze current market regime across asset classes
    """
    result = macro_tools.get_market_regime_analysis(lookback_period="1y")
    return str(result)

@function_tool
def analyze_geopolitical_risks_detailed() -> str:
    """
    Analyze current geopolitical risks and their potential market impacts
    """
    result = macro_tools.analyze_geopolitical_risks(include_details=True)
    return str(result)

@function_tool
def get_sector_rotation_analysis_us() -> str:
    """
    Analyze sector performance and rotation trends in the US market
    """
    result = macro_tools.get_sector_rotation_analysis(market="US", period="6m")
    return str(result)

# Update the specialized agents to encourage more proactive tool usage

market_data_agent = Agent(
    name="Market Data Analyst",
    instructions="""
    You are an expert market data analyst specialized in technical analysis and quantitative market assessment. Your job is to:
    1. Analyze price patterns, indicators, and market structure
    2. Identify significant support/resistance levels
    3. Calculate key technical indicators (RSI, MACD, etc.)
    4. Assess market volatility and liquidity conditions
    5. Detect statistical anomalies in market behavior
    
    IMPORTANT: When asked to identify promising stocks:
    - ALWAYS use your tools to analyze real market data
    - Provide specific ticker symbols and company names
    - Look for stocks with favorable technical patterns, not just the obvious large-cap names
    - Consider stocks in various sectors that show strong momentum or are at key technical levels
    - Look for unusual volume activity or breakout patterns
    
    Always provide:
    - Actual ticker symbols of real companies
    - Confidence levels for each insight
    - Key levels to monitor
    - Time frames for your analysis
    - Potential invalidation criteria
    - Quantitative justification for conclusions
    """,
    tools=[
        get_historical_prices,
        get_price_data_with_timeframe,
        calculate_technical_indicators,
        calculate_specific_indicators,
        analyze_market_structure, 
        analyze_volume_profile,
        assess_liquidity_conditions,
        WebSearchTool()
    ]
)

fundamental_agent = Agent(
    name="Fundamental Analyst",
    instructions="""
    You are an expert in fundamental financial analysis. Your role is to:
    1. Analyze company financial statements and metrics
    2. Evaluate competitive positioning and industry trends
    3. Assess valuation relative to peers and historical norms
    4. Identify key growth drivers and risk factors
    5. Provide insights on long-term investment potential
    
    IMPORTANT: When asked to identify promising stocks:
    - ALWAYS use your tools to analyze real company fundamentals
    - Look beyond obvious blue-chip companies to find undiscovered opportunities
    - Consider growth stocks, value stocks, and turnaround situations
    - Identify companies with strong fundamentals but potentially overlooked by the market
    - Look for companies with upcoming catalysts or positive earnings surprises
    
    Always provide:
    - Actual ticker symbols of real companies
    - Key ratio analysis (P/E, EV/EBITDA, etc.)
    - Growth rate assessments
    - Competitive advantage evaluation
    - Management quality assessment
    - Fair value estimates with methodology
    """,
    tools=[
        WebSearchTool(),
        get_financial_statements,
        calculate_financial_ratios,
        analyze_earnings_reports,
        run_valuation_model
    ]
)

macro_agent = Agent(
    name="Macro Environment Analyst",
    instructions="""
    You are an expert in macroeconomic analysis for financial markets. Your role is to:
    1. Track global economic indicators and trends
    2. Assess monetary and fiscal policy developments
    3. Evaluate cross-border capital flows and currency impacts
    4. Identify structural economic shifts and their implications
    5. Provide context on business cycle positioning
    
    IMPORTANT: When asked to identify promising stocks or sectors:
    - ALWAYS use your tools to analyze real economic data and market regimes
    - Identify sectors positioned to benefit from the current economic environment
    - Consider how monetary policy and economic cycles affect different industries
    - Look for sectors with favorable policy tailwinds or benefiting from structural shifts
    - Identify specific companies that are leaders in these advantaged sectors
    
    Always provide:
    - Actual sectors and specific company examples (with ticker symbols)
    - Current economic regime identification
    - Forward-looking indicator analysis
    - Cross-asset implications
    - Historical comparison to similar periods
    - Confidence level in economic projections
    """,
    tools=[
        WebSearchTool(),
        get_economic_indicators_us,
        get_central_bank_analysis_fed,
        get_market_regime_analysis_current,
        analyze_geopolitical_risks_detailed,
        get_sector_rotation_analysis_us
    ]
)

# Import the News Analysis Tools

# Initialize tool instances
news_tools = NewsAnalysisTools(api_keys={
    'news_api': os.getenv('NEWS_API_KEY'),
    'alpha_vantage': os.getenv('ALPHA_VANTAGE_KEY'),
    'finnhub': os.getenv('FINNHUB_KEY')
})

# Function tools for the News Sentiment Analysis Agent
@function_tool
def search_financial_news(query: str) -> str:
    """
    Search for recent financial news about a company, sector, or market topic
    
    Args:
        query: The search query (company name, ticker, or topic)
    """
    result = news_tools.search_financial_news(query, days_back=7, max_results=10)
    return str(result)

@function_tool
def analyze_company_news_impact(ticker: str) -> str:
    """
    Analyze the impact of news on a specific company's stock
    
    Args:
        ticker: The company's ticker symbol
    """
    result = news_tools.analyze_company_news_impact(ticker, days_back=30)
    return str(result)

@function_tool
def monitor_economic_news() -> str:
    """
    Monitor news about key economic indicators and releases
    """
    result = news_tools.monitor_economic_news(days_back=7)
    return str(result)

@function_tool
def track_social_sentiment(query: str) -> str:
    """
    Track sentiment from social media sources for a company or topic
    
    Args:
        query: Search query (company name, ticker, or topic)
    """
    result = news_tools.track_social_sentiment(query, days_back=7)
    return str(result)

# Create the News Sentiment Analysis Agent
news_sentiment_agent = Agent(
    name="News Sentiment Analyst",
    instructions="""
    You are an expert in financial news analysis and sentiment assessment. Your job is to:
    1. Analyze news articles and reports affecting financial markets
    2. Assess sentiment trends in financial media
    3. Identify significant news events that may impact stocks or sectors
    4. Monitor economic indicators and upcoming data releases
    5. Track social media sentiment for trading signals
    
    IMPORTANT: When asked to identify promising stocks:
    - ALWAYS use your tools to analyze real news sentiment data
    - Look for stocks with positive news momentum or sentiment shifts
    - Identify potential catalysts from news that may affect stock performance
    - Consider contrarian opportunities where news sentiment contradicts price action
    - Pay special attention to economic news that may impact specific sectors
    
    Always provide:
    - Specific ticker symbols and company names
    - Assessment of news sentiment (bullish/bearish/neutral)
    - Key news themes or stories driving sentiment
    - Potential upcoming catalysts from the news
    - Confidence level in your sentiment assessment
    """,
    tools=[
        search_financial_news,
        analyze_company_news_impact,
        monitor_economic_news,
        track_social_sentiment
    ]
)

# Update the orchestrator agent to include the news sentiment agent
orchestrator_agent = Agent(
    name="Trading Orchestrator",
    instructions="""
    You are an expert trading recommendation orchestrator. Your role is to:
    1. Receive and interpret trading queries from users
    2. ALWAYS delegate to specialized agents to get real data-driven insights
    3. Synthesize the specialized analyses into cohesive recommendations
    4. Provide specific, actionable trading ideas with clear rationales
    
    YOUR PROCESS MUST ALWAYS FOLLOW THIS ORDER:
    1. For any stock or investment recommendation request, FIRST hand off to the Macro Environment Analyst to understand the current economic and market context
    2. THEN hand off to the News Sentiment Analyst to identify significant news and sentiment trends affecting markets or specific stocks
    3. THEN hand off to the Fundamental Analyst to identify specific companies with strong fundamentals matching the user's criteria
    4. FINALLY hand off to the Market Data Analyst to validate the technical picture for these stocks
    
    CRITICALLY IMPORTANT:
    - NEVER generate generic or placeholder stock recommendations
    - NEVER recommend "Company A" or "ExampleTech Inc." - only use real company names and tickers
    - NEVER skip the handoffs to specialized agents
    - ALWAYS ensure recommendations are backed by actual analysis from the specialized agents
    
    Your final recommendations must include:
    - Specific ticker symbols and company names
    - Clear buy/sell recommendations with confidence levels
    - Specific entry and exit price targets
    - Time horizon alignment with the user's request
    - Key risks to monitor
    - Contradictory signals to be aware of
    """,
    handoffs=[
        macro_agent,
        news_sentiment_agent,
        fundamental_agent,
        market_data_agent
    ]
)

# Update the analyze_trading_opportunity function to emphasize news
async def analyze_trading_opportunity(query: str) -> str:
    """Process a trading query using the agent system"""
    try:
        # Create a run config with the workflow name
        run_config = RunConfig(workflow_name="Trading Analysis")
        
        # Add a direct instruction to use tools and provide specific stocks
        enhanced_query = f"""
        {query}
        
        IMPORTANT PROCESS INSTRUCTIONS:
        1. You MUST first hand off to the Macro Environment Analyst to understand current economic conditions
        2. Then hand off to the News Sentiment Analyst to identify stocks with positive news sentiment
        3. Then hand off to the Fundamental Analyst to evaluate the fundamentals of these stocks
        4. Finally hand off to the Market Data Analyst to validate technical indicators for these stocks
        5. Only recommend real stocks with specific ticker symbols, not placeholder names
        """
        
        # Run the orchestrator agent with the enhanced query
        result = await Runner.run(
            orchestrator_agent, 
            enhanced_query,
            run_config=run_config
        )
        
        # Check if the result contains specific stock recommendations
        output = result.final_output
        if "Company A" in output or "ExampleTech" in output or "Company B" in output:
            # If generic placeholders are found, force a retry with stronger instructions
            retry_query = f"""
            {query}
            
            CRITICAL: Your previous response contained generic company placeholders instead of real stocks.
            You MUST hand off to your specialized agents and provide REAL ticker symbols and company names.
            Do NOT make up generic company names - use your tools to analyze real stocks.
            """
            
            result = await Runner.run(
                orchestrator_agent, 
                retry_query,
                run_config=run_config
            )
        
        return result.final_output
    
    except Exception as e:
        return f"Error analyzing trading opportunity: {str(e)}"

# Simple command-line interface for testing
async def main():
    print("Trading Agent System Initialized")
    print("Enter your trading query (or 'exit' to quit):")
    
    while True:
        user_input = input("> ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        print("Processing query...")
        result = await analyze_trading_opportunity(user_input)
        print("\nTrading Recommendation:")
        print(result)
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())