import os
import logging
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

class MacroEnvironmentTools:
    """Implementation of tools for the macro environment analysis agent"""
    
    def __init__(self, api_keys=None):
        """Initialize with necessary API keys"""
        self.api_keys = api_keys or {}
        self.fred_api_key = self.api_keys.get('fred', '')
        self.world_bank_api_key = self.api_keys.get('world_bank', '')
        self.alpha_vantage_key = self.api_keys.get('alpha_vantage', '')
        
        # Cache for expensive API calls
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 24 * 60 * 60  # 24 hours in seconds
    
    def get_economic_indicators(self, indicators=None, country='US', period='5y'):
        """
        Retrieve key economic indicators like GDP, inflation, unemployment, etc.
        
        Args:
            indicators (list): List of indicators to retrieve (e.g., ['gdp', 'inflation', 'unemployment'])
            country (str): Country code for the data
            period (str): Time period (e.g., '5y', '10y', 'max')
            
        Returns:
            dict: Economic indicator data with time series
        """
        try:
            # Default indicators if none specified
            if not indicators:
                indicators = ['gdp', 'inflation', 'unemployment', 'interest_rates', 'retail_sales']
            
            # Map indicator names to FRED series IDs
            indicator_map = {
                'gdp': 'GDP',                 # Real Gross Domestic Product
                'inflation': 'CPIAUCSL',      # Consumer Price Index for All Urban Consumers
                'core_inflation': 'CPILFESL', # CPI Less Food and Energy
                'unemployment': 'UNRATE',     # Unemployment Rate
                'interest_rates': 'FEDFUNDS', # Federal Funds Rate
                'retail_sales': 'RSAFS',      # Retail Sales
                'industrial_production': 'INDPRO', # Industrial Production Index
                'consumer_sentiment': 'UMCSENT', # University of Michigan Consumer Sentiment
                'housing_starts': 'HOUST',    # Housing Starts
                'home_prices': 'CSUSHPINSA',  # Case-Shiller Home Price Index
                'trade_balance': 'NETEXP',    # Net Exports of Goods and Services
                'government_debt': 'GFDEBTN', # Federal Debt: Total Public Debt
                'yield_curve': 'T10Y2Y'       # 10-Year Treasury Constant Maturity Minus 2-Year
            }
            
            # Parse period to determine start date
            end_date = datetime.now()
            if period == '1y':
                start_date = end_date - relativedelta(years=1)
            elif period == '5y':
                start_date = end_date - relativedelta(years=5)
            elif period == '10y':
                start_date = end_date - relativedelta(years=10)
            elif period == 'max':
                start_date = datetime(1950, 1, 1)  # Arbitrary early date
            else:
                # Default to 5 years
                start_date = end_date - relativedelta(years=5)
            
            # Format dates for FRED API
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            results = {}
            
            # For each requested indicator, fetch data from FRED
            for indicator in indicators:
                if indicator in indicator_map:
                    series_id = indicator_map[indicator]
                    
                    # Check cache first
                    cache_key = f"{indicator}_{country}_{period}"
                    if cache_key in self.cache and datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
                        results[indicator] = self.cache[cache_key]
                        continue
                    
                    # Construct FRED API URL
                    url = f"https://api.stlouisfed.org/fred/series/observations"
                    params = {
                        "series_id": series_id,
                        "api_key": self.fred_api_key,
                        "file_type": "json",
                        "observation_start": start_date_str,
                        "observation_end": end_date_str,
                        "frequency": "m"  # Monthly frequency
                    }
                    
                    # Fetch data
                    if self.fred_api_key:
                        response = requests.get(url, params=params)
                        if response.status_code == 200:
                            data = response.json()
                            
                            # Process observations
                            observations = data.get('observations', [])
                            time_series = {obs['date']: float(obs['value']) if obs['value'] != '.' else None for obs in observations}
                            
                            # Calculate latest value, change, and percent change
                            values = [v for v in time_series.values() if v is not None]
                            latest_value = values[-1] if values else None
                            previous_value = values[-2] if len(values) >= 2 else None
                            
                            change = latest_value - previous_value if latest_value is not None and previous_value is not None else None
                            percent_change = (change / previous_value * 100) if change is not None and previous_value is not None else None
                            
                            indicator_data = {
                                "latest_value": latest_value,
                                "change": change,
                                "percent_change": percent_change,
                                "series": time_series
                            }
                            
                            # Cache the result
                            self.cache[cache_key] = indicator_data
                            self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_duration
                            
                            results[indicator] = indicator_data
                        else:
                            results[indicator] = {
                                "error": f"FRED API Error: {response.status_code}",
                                "message": response.text
                            }
                    else:
                        # Simulation mode (no API key)
                        # Generate synthetic data for demo purposes
                        dates = pd.date_range(start=start_date, end=end_date, freq='M')
                        
                        # Different simulated patterns based on indicator
                        if indicator == 'gdp':
                            # Simulated GDP growth with seasonal pattern
                            values = np.linspace(20000, 25000, len(dates))
                            values = values + np.sin(np.arange(len(dates)) * 0.5) * 200  # Seasonal component
                            values = values * (1 + np.random.normal(0, 0.01, len(dates)))  # Random noise
                        elif indicator == 'inflation':
                            # Simulated inflation fluctuating between 1-4%
                            values = 2 + np.sin(np.arange(len(dates)) * 0.2) * 1.5
                            values = values + np.random.normal(0, 0.2, len(dates))
                        elif indicator == 'unemployment':
                            # Simulated unemployment starting high and trending down
                            values = np.linspace(6, 3.5, len(dates))
                            values = values + np.random.normal(0, 0.2, len(dates))
                        elif indicator == 'interest_rates':
                            # Simulated interest rates with an upward trend
                            values = np.linspace(0.25, 4.75, len(dates))
                            values = values + np.random.normal(0, 0.15, len(dates))
                            values = np.maximum(0, values)  # Ensure non-negative
                        else:
                            # Generic simulated data with upward trend and noise
                            values = np.linspace(100, 130, len(dates))
                            values = values * (1 + np.random.normal(0, 0.03, len(dates)))
                        
                        # Convert to time series format
                        time_series = {date.strftime('%Y-%m-%d'): round(value, 2) for date, value in zip(dates, values)}
                        
                        # Calculate metrics
                        latest_value = round(values[-1], 2)
                        previous_value = round(values[-2], 2) if len(values) >= 2 else None
                        change = round(latest_value - previous_value, 2) if previous_value is not None else None
                        percent_change = round((change / previous_value * 100), 2) if change is not None and previous_value is not None else None
                        
                        indicator_data = {
                            "latest_value": latest_value,
                            "change": change,
                            "percent_change": percent_change,
                            "series": time_series,
                            "note": "Simulated data (no FRED API key provided)"
                        }
                        
                        # Cache the result
                        self.cache[cache_key] = indicator_data
                        self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_duration
                        
                        results[indicator] = indicator_data
                else:
                    results[indicator] = {
                        "error": "Unknown indicator",
                        "message": f"Indicator '{indicator}' not recognized"
                    }
            
            # Add metadata
            metadata = {
                "country": country,
                "period": period,
                "data_source": "FRED" if self.fred_api_key else "Simulated data",
                "as_of_date": datetime.now().strftime('%Y-%m-%d')
            }
            
            # Economic cycle analysis
            cycle_analysis = self._analyze_economic_cycle(results)
            
            return {
                "status": "success",
                "metadata": metadata,
                "indicators": results,
                "economic_cycle": cycle_analysis
            }
            
        except Exception as e:
            logger.error(f"Error retrieving economic indicators: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to retrieve economic indicators: {str(e)}"
            }
    
    def _analyze_economic_cycle(self, indicators):
        """
        Analyze the current economic cycle based on indicator patterns
        
        Args:
            indicators (dict): Economic indicators data
            
        Returns:
            dict: Economic cycle analysis
        """
        # Simple rule-based analysis
        # In a real implementation, this would be more sophisticated
        
        cycle_indicators = {}
        
        # Analyze GDP trend
        if 'gdp' in indicators and indicators['gdp'].get('percent_change') is not None:
            gdp_growth = indicators['gdp']['percent_change']
            if gdp_growth > 3:
                cycle_indicators['gdp'] = "expansion"
            elif gdp_growth > 0:
                cycle_indicators['gdp'] = "moderate growth"
            elif gdp_growth > -1:
                cycle_indicators['gdp'] = "slowdown"
            else:
                cycle_indicators['gdp'] = "contraction"
        
        # Analyze unemployment trend
        if 'unemployment' in indicators and indicators['unemployment'].get('change') is not None:
            unemployment_change = indicators['unemployment']['change']
            if unemployment_change < -0.3:
                cycle_indicators['unemployment'] = "improving strongly"
            elif unemployment_change < 0:
                cycle_indicators['unemployment'] = "improving moderately"
            elif unemployment_change < 0.2:
                cycle_indicators['unemployment'] = "stable"
            else:
                cycle_indicators['unemployment'] = "deteriorating"
        
        # Analyze inflation
        if 'inflation' in indicators and indicators['inflation'].get('latest_value') is not None:
            inflation_rate = indicators['inflation']['latest_value']
            if inflation_rate > 5:
                cycle_indicators['inflation'] = "high"
            elif inflation_rate > 3:
                cycle_indicators['inflation'] = "elevated"
            elif inflation_rate > 1:
                cycle_indicators['inflation'] = "moderate"
            else:
                cycle_indicators['inflation'] = "low"
        
        # Analyze interest rates
        if 'interest_rates' in indicators and indicators['interest_rates'].get('latest_value') is not None:
            interest_rate = indicators['interest_rates']['latest_value']
            if interest_rate > 4:
                cycle_indicators['monetary_policy'] = "restrictive"
            elif interest_rate > 2:
                cycle_indicators['monetary_policy'] = "neutral"
            else:
                cycle_indicators['monetary_policy'] = "accommodative"
        
        # Determine overall cycle phase (simplified)
        if 'gdp' in cycle_indicators and 'inflation' in cycle_indicators:
            gdp_indicator = cycle_indicators['gdp']
            inflation_indicator = cycle_indicators['inflation']
            
            if gdp_indicator in ['expansion'] and inflation_indicator in ['low', 'moderate']:
                current_phase = "mid-cycle expansion"
            elif gdp_indicator in ['expansion'] and inflation_indicator in ['elevated', 'high']:
                current_phase = "late-cycle expansion"
            elif gdp_indicator in ['moderate growth'] and inflation_indicator in ['low', 'moderate']:
                current_phase = "early-cycle expansion"
            elif gdp_indicator in ['slowdown', 'contraction']:
                current_phase = "contraction/recession"
            else:
                current_phase = "transitional"
        else:
            current_phase = "undetermined (insufficient data)"
        
        return {
            "current_phase": current_phase,
            "indicators": cycle_indicators,
            "analysis_methodology": "Simplified rule-based methodology"
        }
    
    def get_central_bank_analysis(self, central_bank="federal_reserve", lookback_months=6):
        """
        Analyze central bank communications and policy trends
        
        Args:
            central_bank (str): Central bank to analyze (e.g., "federal_reserve", "ecb", "boj")
            lookback_months (int): Number of months to look back
            
        Returns:
            dict: Central bank policy analysis
        """
        try:
            # Calculate start date for lookback period
            end_date = datetime.now()
            start_date = end_date - relativedelta(months=lookback_months)
            
            # For a real implementation, you would:
            # 1. Scrape/access central bank press releases
            # 2. Analyze policy statements for sentiment
            # 3. Track rate decisions and guidance
            
            # Map of central banks
            central_bank_map = {
                "federal_reserve": "Federal Reserve (US)",
                "ecb": "European Central Bank",
                "boj": "Bank of Japan",
                "boe": "Bank of England",
                "pboc": "People's Bank of China",
                "rba": "Reserve Bank of Australia",
                "rbi": "Reserve Bank of India",
                "cbr": "Central Bank of Russia"
            }
            
            bank_name = central_bank_map.get(central_bank, central_bank)
            
            # Check cache
            cache_key = f"central_bank_{central_bank}_{lookback_months}"
            if cache_key in self.cache and datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
                return self.cache[cache_key]
            
            # Simulated data for demonstration
            # In a real implementation, this would come from API calls or web scraping
            
            # Simulated policy rate changes
            if central_bank == "federal_reserve":
                latest_rate = 5.25  # Current Fed Funds rate as of example time
                rate_changes = [
                    {"date": (end_date - relativedelta(months=0)).strftime('%Y-%m-%d'), "rate": 5.25, "change": 0.0, "statement_sentiment": "hawkish"},
                    {"date": (end_date - relativedelta(months=1)).strftime('%Y-%m-%d'), "rate": 5.25, "change": 0.0, "statement_sentiment": "neutral"},
                    {"date": (end_date - relativedelta(months=2)).strftime('%Y-%m-%d'), "rate": 5.25, "change": 0.0, "statement_sentiment": "neutral"},
                    {"date": (end_date - relativedelta(months=3)).strftime('%Y-%m-%d'), "rate": 5.25, "change": 0.25, "statement_sentiment": "hawkish"},
                    {"date": (end_date - relativedelta(months=4)).strftime('%Y-%m-%d'), "rate": 5.0, "change": 0.25, "statement_sentiment": "hawkish"},
                    {"date": (end_date - relativedelta(months=5)).strftime('%Y-%m-%d'), "rate": 4.75, "change": 0, "statement_sentiment": "hawkish"}
                ]
                
                policy_focus = ["inflation control", "labor market stability", "financial market risks"]
                outlook = "The Federal Reserve is maintaining a restrictive policy stance to ensure inflation returns to 2%. Recent communications indicate caution about premature rate cuts while acknowledging progress on inflation."
                key_quotes = [
                    "We are committed to bringing inflation back to our 2 percent goal.",
                    "Recent data shows progress on inflation, but we need to see sustained improvement.",
                    "The labor market remains tight but has shown signs of cooling."
                ]
            elif central_bank == "ecb":
                latest_rate = 4.0
                rate_changes = [
                    {"date": (end_date - relativedelta(months=0)).strftime('%Y-%m-%d'), "rate": 4.0, "change": 0.0, "statement_sentiment": "neutral"},
                    {"date": (end_date - relativedelta(months=1)).strftime('%Y-%m-%d'), "rate": 4.0, "change": 0.0, "statement_sentiment": "neutral"},
                    {"date": (end_date - relativedelta(months=2)).strftime('%Y-%m-%d'), "rate": 4.0, "change": 0.25, "statement_sentiment": "hawkish"},
                    {"date": (end_date - relativedelta(months=3)).strftime('%Y-%m-%d'), "rate": 3.75, "change": 0.25, "statement_sentiment": "hawkish"},
                    {"date": (end_date - relativedelta(months=4)).strftime('%Y-%m-%d'), "rate": 3.5, "change": 0.0, "statement_sentiment": "hawkish"},
                    {"date": (end_date - relativedelta(months=5)).strftime('%Y-%m-%d'), "rate": 3.5, "change": 0.25, "statement_sentiment": "hawkish"}
                ]
                
                policy_focus = ["inflation control", "economic growth concerns", "regional economic divergence"]
                outlook = "The ECB has been focused on addressing persistent inflation while monitoring signs of economic slowdown in key eurozone economies."
                key_quotes = [
                    "We will keep interest rates sufficiently restrictive for as long as necessary to achieve a timely return of inflation to our target.",
                    "Recent data points to weaker growth dynamics in some eurozone economies.",
                    "Inflation remains above our target but is on a downward path."
                ]
            else:
                # Generic simulation for other central banks
                latest_rate = 3.0
                rate_changes = [
                    {"date": (end_date - relativedelta(months=i)).strftime('%Y-%m-%d'), 
                     "rate": 3.0 - (i * 0.25 if i <= 3 else 0.75), 
                     "change": 0.25 if i <= 3 and i % 2 == 0 else 0.0,
                     "statement_sentiment": "neutral" if i % 2 == 0 else "hawkish"} 
                    for i in range(lookback_months)
                ]
                
                policy_focus = ["inflation control", "economic stability"]
                outlook = f"The {bank_name} has been adjusting policy to balance inflation concerns with economic growth."
                key_quotes = [
                    "Our monetary policy remains data-dependent.",
                    "We are committed to price stability while supporting economic growth."
                ]
            
            # Calculate policy trend
            rate_trend = "stable"
            sentiment_trend = "neutral"
            
            if len(rate_changes) >= 2:
                first_rate = rate_changes[-1]["rate"]
                latest_rate = rate_changes[0]["rate"]
                
                rate_diff = latest_rate - first_rate
                if rate_diff > 0.5:
                    rate_trend = "strongly tightening"
                elif rate_diff > 0:
                    rate_trend = "mildly tightening"
                elif rate_diff < -0.5:
                    rate_trend = "strongly easing"
                elif rate_diff < 0:
                    rate_trend = "mildly easing"
                
                # Analyze sentiment trend
                hawkish_count = sum(1 for change in rate_changes if change["statement_sentiment"] == "hawkish")
                dovish_count = sum(1 for change in rate_changes if change["statement_sentiment"] == "dovish")
                
                if hawkish_count > len(rate_changes) / 2:
                    sentiment_trend = "predominantly hawkish"
                elif dovish_count > len(rate_changes) / 2:
                    sentiment_trend = "predominantly dovish"
                else:
                    sentiment_trend = "mixed or neutral"
            
            # Simulate forward guidance
            if rate_trend in ["strongly tightening", "mildly tightening"] and sentiment_trend == "predominantly hawkish":
                forward_guidance = "Continued policy tightening likely"
            elif rate_trend in ["strongly easing", "mildly easing"] and sentiment_trend == "predominantly dovish":
                forward_guidance = "Further policy easing expected"
            elif rate_trend == "stable" and sentiment_trend in ["mixed or neutral", "predominantly hawkish"]:
                forward_guidance = "Holding rates steady with potential for additional tightening if needed"
            elif rate_trend == "stable" and sentiment_trend == "predominantly dovish":
                forward_guidance = "Holding rates steady with bias toward future cuts"
            else:
                forward_guidance = "Policy direction uncertain, dependent on incoming data"
            
            result = {
                "status": "success",
                "central_bank": bank_name,
                "latest_policy_rate": latest_rate,
                "policy_changes": rate_changes,
                "analysis": {
                    "rate_trend": rate_trend,
                    "sentiment_trend": sentiment_trend,
                    "policy_focus": policy_focus,
                    "forward_guidance": forward_guidance,
                    "outlook": outlook,
                    "key_quotes": key_quotes
                },
                "data_source": "Simulated data - would use central bank communications in production"
            }
            
            # Cache the result
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_duration
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing central bank policy: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to analyze central bank policy: {str(e)}"
            }
    
    def get_market_regime_analysis(self, lookback_period='1y'):
        """
        Analyze current market regime across asset classes
        
        Args:
            lookback_period (str): Period to analyze ('1m', '3m', '6m', '1y', '3y')
            
        Returns:
            dict: Market regime analysis including correlations, volatility, and trend strength
        """
        try:
            # Define major asset class proxies (ETF tickers)
            asset_classes = {
                'us_equities': 'SPY',         # S&P 500
                'developed_intl': 'EFA',      # Developed markets ex-US
                'emerging_markets': 'EEM',    # Emerging markets
                'us_treasury': 'IEF',         # 7-10 year Treasury
                'corporate_bonds': 'LQD',     # Investment grade corporate bonds
                'high_yield': 'HYG',          # High yield bonds
                'gold': 'GLD',                # Gold
                'commodities': 'DBC',         # Commodities
                'real_estate': 'VNQ',         # Real estate
                'dollar_index': 'UUP'         # US Dollar
            }
            
            # Check cache
            cache_key = f"market_regime_{lookback_period}"
            if cache_key in self.cache and datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
                return self.cache[cache_key]
            
            # Convert lookback period to days for yfinance
            period_map = {
                '1m': '30d',
                '3m': '90d',
                '6m': '180d',
                '1y': '365d',
                '3y': '1095d'
            }
            yf_period = period_map.get(lookback_period, '365d')
            
            # Fetch price data for all asset classes
            price_data = {}
            returns_data = {}
            
            for asset_name, ticker in asset_classes.items():
                try:
                    # Get historical data
                    asset = yf.Ticker(ticker)
                    history = asset.history(period=yf_period)
                    
                    if not history.empty:
                        # Store adjusted close prices
                        price_data[asset_name] = history['Close'].copy()
                        
                        # Calculate returns
                        returns_data[asset_name] = history['Close'].pct_change().dropna()
                    else:
                        logger.warning(f"No data retrieved for {asset_name} ({ticker})")
                except Exception as asset_error:
                    logger.warning(f"Error fetching data for {asset_name} ({ticker}): {str(asset_error)}")
            
            # Convert price and returns data to DataFrames
            if price_data:
                prices_df = pd.DataFrame(price_data)
                returns_df = pd.DataFrame(returns_data)
                
                # Calculate performance metrics
                performance = {}
                volatility = {}
                for asset in returns_df.columns:
                    # Total return over period
                    first_price = prices_df[asset].iloc[0]
                    last_price = prices_df[asset].iloc[-1]
                    total_return = (last_price / first_price - 1) * 100
                    
                    # Annualize return
                    days = (prices_df.index[-1] - prices_df.index[0]).days
                    annual_return = ((1 + total_return/100) ** (365/days) - 1) * 100 if days > 0 else 0
                    
                    # Calculate volatility (annualized standard deviation)
                    asset_volatility = returns_df[asset].std() * (252 ** 0.5) * 100  # Annualize by multiplying by sqrt(252)
                    
                    # Maximum drawdown
                    cumulative = (1 + returns_df[asset]).cumprod()
                    running_max = cumulative.cummax()
                    drawdown = (cumulative / running_max - 1) * 100
                    max_drawdown = drawdown.min()
                    
                    # Risk-adjusted return (Sharpe Ratio) - assuming 0% risk-free rate for simplicity
                    sharpe = annual_return / asset_volatility if asset_volatility > 0 else 0
                    
                    performance[asset] = {
                        'total_return_pct': round(total_return, 2),
                        'annual_return_pct': round(annual_return, 2),
                        'max_drawdown_pct': round(max_drawdown, 2),
                        'sharpe_ratio': round(sharpe, 2)
                    }
                    
                    volatility[asset] = round(asset_volatility, 2)
                
                # Calculate correlation matrix
                correlation_matrix = returns_df.corr().round(2)
                
                # Identify current market regime
                # Simplified approach - in a real system this would be more sophisticated
                
                # 1. Check equity-bond correlation
                if 'us_equities' in correlation_matrix.columns and 'us_treasury' in correlation_matrix.columns:
                    equity_bond_corr = correlation_matrix.loc['us_equities', 'us_treasury']
                    
                    if equity_bond_corr < -0.3:
                        regime_type = "Traditional (Negative Equity-Bond Correlation)"
                    elif equity_bond_corr > 0.3:
                        regime_type = "Inflationary (Positive Equity-Bond Correlation)"
                    else:
                        regime_type = "Transitional"
                else:
                    regime_type = "Undetermined (Missing Data)"
                
                # 2. Check volatility regime
                if 'us_equities' in volatility:
                    equity_vol = volatility['us_equities']
                    
                    if equity_vol > 25:
                        vol_regime = "High Volatility"
                    elif equity_vol > 15:
                        vol_regime = "Moderate Volatility"
                    else:
                        vol_regime = "Low Volatility"
                else:
                    vol_regime = "Undetermined"
                
                # 3. Check performance trends
                risk_on_assets = ['us_equities', 'emerging_markets', 'high_yield']
                risk_off_assets = ['us_treasury', 'gold', 'dollar_index']
                
                risk_on_perf = [performance[asset]['total_return_pct'] for asset in risk_on_assets if asset in performance]
                risk_off_perf = [performance[asset]['total_return_pct'] for asset in risk_off_assets if asset in performance]
                
                avg_risk_on = sum(risk_on_perf) / len(risk_on_perf) if risk_on_perf else None
                avg_risk_off = sum(risk_off_perf) / len(risk_off_perf) if risk_off_perf else None
                
                if avg_risk_on is not None and avg_risk_off is not None:
                    if avg_risk_on > 10 and avg_risk_on > avg_risk_off:
                        risk_regime = "Risk-On"
                    elif avg_risk_off > 5 and avg_risk_off > avg_risk_on:
                        risk_regime = "Risk-Off"
                    else:
                        risk_regime = "Neutral"
                else:
                    risk_regime = "Undetermined"
                
                # Combined regime assessment
                regime_assessment = f"{regime_type} | {vol_regime} | {risk_regime}"
                
                # Generate implications for asset classes
                implications = {
                    "favorable_assets": [],
                    "unfavorable_assets": [],
                    "rationale": ""
                }
                
                if regime_type == "Traditional (Negative Equity-Bond Correlation)":
                    if risk_regime == "Risk-On":
                        implications["favorable_assets"] = ["Equities", "High Yield Bonds", "Commodities"]
                        implications["unfavorable_assets"] = ["Treasury Bonds", "Cash", "Defensive Sectors"]
                        implications["rationale"] = "In a traditional risk-on environment, growth assets typically outperform."
                    else:
                        implications["favorable_assets"] = ["Treasury Bonds", "Investment Grade Bonds", "Defensive Equities"]
                        implications["unfavorable_assets"] = ["Cyclical Equities", "High Yield Bonds", "Commodities"]
                        implications["rationale"] = "In a traditional risk-off environment, safe-haven assets typically outperform."
                elif regime_type == "Inflationary (Positive Equity-Bond Correlation)":
                    implications["favorable_assets"] = ["Commodities", "TIPS", "Value Stocks", "Real Assets"]
                    implications["unfavorable_assets"] = ["Long Duration Bonds", "Growth Stocks"]
                    implications["rationale"] = "In an inflationary regime, real assets and inflation hedges tend to outperform nominal assets."
                
                if vol_regime == "High Volatility":
                    implications["favorable_assets"].extend(["Low Volatility Stocks", "Cash", "Managed Futures"])
                    implications["unfavorable_assets"].extend(["Small Caps", "Emerging Markets"])
                    implications["rationale"] += " High volatility favors defensive positioning and strategies that benefit from market dislocations."
                
                # Remove duplicates
                implications["favorable_assets"] = list(set(implications["favorable_assets"]))
                implications["unfavorable_assets"] = list(set(implications["unfavorable_assets"]))
                
                result = {
                    "status": "success",
                    "lookback_period": lookback_period,
                    "market_regime": {
                        "overall_assessment": regime_assessment,
                        "correlation_regime": regime_type,
                        "volatility_regime": vol_regime,
                        "risk_sentiment": risk_regime
                    },
                    "performance": performance,
                    "volatility": volatility,
                    "correlation_matrix": correlation_matrix.to_dict(),
                    "implications": implications,
                    "data_as_of": prices_df.index[-1].strftime('%Y-%m-%d')
                }
            else:
                result = {
                    "status": "error",
                    "message": "No price data could be retrieved for asset classes"
                }
            
            # Cache the result
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_duration
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to analyze market regime: {str(e)}"
            }
    
    def analyze_geopolitical_risks(self, include_details=True):
        """
        Analyze current geopolitical risks and their potential market impacts
        
        Args:
            include_details (bool): Whether to include detailed descriptions
            
        Returns:
            dict: Geopolitical risk analysis
        """
        try:
            # In a real implementation, this would use:
            # - Geopolitical risk indices
            # - News sentiment analysis
            # - Expert assessments
            
            # For demonstration, we'll use simulated data
            
            # Check cache
            cache_key = f"geopolitical_risks_{include_details}"
            if cache_key in self.cache and datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
                return self.cache[cache_key]
            
            # Current date for the analysis
            analysis_date = datetime.now().strftime('%Y-%m-%d')
            
            # Simulated geopolitical risks
            risks = [
                {
                    "risk_category": "International Conflict",
                    "regions_affected": ["Eastern Europe", "Russia", "Ukraine"],
                    "risk_level": "high",
                    "market_impact": "moderate",
                    "affected_assets": ["Energy Commodities", "European Equities", "Defense Stocks", "Russian/Ukrainian Assets"],
                    "description": "Ongoing conflict with potential for escalation and impact on energy markets and global supply chains.",
                    "trend": "stable"
                },
                {
                    "risk_category": "Trade Tensions",
                    "regions_affected": ["United States", "China", "Global"],
                    "risk_level": "moderate",
                    "market_impact": "moderate",
                    "affected_assets": ["Chinese Equities", "US Technology", "Semiconductors", "Global Supply Chains"],
                    "description": "Persistent trade and technology tensions affecting global supply chains and specific sectors.",
                    "trend": "deteriorating"
                },
                {
                    "risk_category": "Middle East Instability",
                    "regions_affected": ["Middle East", "Global Energy Markets"],
                    "risk_level": "high",
                    "market_impact": "moderate",
                    "affected_assets": ["Oil", "Middle Eastern Equities", "Defense Contractors"],
                    "description": "Regional conflicts with potential to disrupt global energy supplies and shipping routes.",
                    "trend": "volatile"
                },
                {
                    "risk_category": "Policy Uncertainty",
                    "regions_affected": ["Global", "Advanced Economies"],
                    "risk_level": "moderate",
                    "market_impact": "low",
                    "affected_assets": ["Government Bonds", "Currencies", "Sensitive Sectors"],
                    "description": "Uncertainty surrounding monetary, fiscal, and regulatory policies in major economies.",
                    "trend": "improving"
                },
                {
                    "risk_category": "Cybersecurity Threats",
                    "regions_affected": ["Global", "Technology Sector"],
                    "risk_level": "moderate",
                    "market_impact": "low",
                    "affected_assets": ["Technology Stocks", "Cybersecurity Companies", "Critical Infrastructure"],
                    "description": "Increasing frequency and sophistication of cyberattacks targeting corporations and infrastructure.",
                    "trend": "deteriorating"
                }
            ]
            
            # If details not requested, remove descriptions
            if not include_details:
                for risk in risks:
                    if 'description' in risk:
                        del risk['description']
            
            # Calculate aggregate risk level
            risk_levels = {"low": 1, "moderate": 2, "high": 3, "severe": 4}
            impact_levels = {"low": 1, "moderate": 2, "high": 3, "severe": 4}
            
            avg_risk = sum(risk_levels.get(risk['risk_level'], 0) for risk in risks) / len(risks)
            avg_impact = sum(impact_levels.get(risk['market_impact'], 0) for risk in risks) / len(risks)
            
            if avg_risk < 1.5:
                overall_risk = "low"
            elif avg_risk < 2.5:
                overall_risk = "moderate"
            elif avg_risk < 3.5:
                overall_risk = "high"
            else:
                overall_risk = "severe"
            
            if avg_impact < 1.5:
                overall_impact = "low"
            elif avg_impact < 2.5:
                overall_impact = "moderate"
            elif avg_impact < 3.5:
                overall_impact = "high"
            else:
                overall_impact = "severe"
            
            # Aggregate trends
            trends = [risk['trend'] for risk in risks]
            trend_counts = {trend: trends.count(trend) for trend in set(trends)}
            
            if trend_counts.get('deteriorating', 0) > len(risks) / 3:
                overall_trend = "deteriorating"
            elif trend_counts.get('improving', 0) > len(risks) / 3:
                overall_trend = "improving"
            elif trend_counts.get('volatile', 0) > len(risks) / 3:
                overall_trend = "volatile"
            else:
                overall_trend = "stable"
            
            # Investment implications
            if overall_risk == "high" and overall_impact in ["moderate", "high"]:
                implications = "Consider reducing exposure to risk assets and increasing allocations to safe havens."
                positioning = ["Reduced Risk Exposure", "Hedging Strategies", "Safe Haven Assets"]
            elif overall_risk == "moderate" and overall_trend == "deteriorating":
                implications = "Maintain balanced exposure with targeted hedges against specific risks."
                positioning = ["Balanced Allocation", "Targeted Hedges", "Quality Focus"]
            elif overall_risk in ["low", "moderate"] and overall_trend in ["stable", "improving"]:
                implications = "Standard diversified allocation with normal risk levels appropriate."
                positioning = ["Strategic Asset Allocation", "Normal Risk Exposure", "Diversification"]
            else:
                implications = "Maintain diversification with some defensive positioning."
                positioning = ["Diversification", "Some Defensive Positioning"]
            
            result = {
                "status": "success",
                "analysis_date": analysis_date,
                "geopolitical_risk_assessment": {
                    "overall_risk_level": overall_risk,
                    "market_impact": overall_impact,
                    "trend": overall_trend
                },
                "specific_risks": risks,
                "investment_implications": {
                    "summary": implications,
                    "recommended_positioning": positioning
                },
                "data_source": "Simulated analysis - would use news sentiment, risk indices in production"
            }
            
            # Cache the result
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_duration
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing geopolitical risks: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to analyze geopolitical risks: {str(e)}"
            }
    
    def get_sector_rotation_analysis(self, market='US', period='6m'):
        """
        Analyze sector performance and rotation trends
        
        Args:
            market (str): Market to analyze (US, Europe, Global, etc.)
            period (str): Time period to analyze (1m, 3m, 6m, 1y, 3y)
            
        Returns:
            dict: Sector rotation analysis
        """
        try:
            # Check cache
            cache_key = f"sector_rotation_{market}_{period}"
            if cache_key in self.cache and datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
                return self.cache[cache_key]
            
            # Define sector ETFs by market
            sector_etfs = {}
            
            if market == 'US':
                sector_etfs = {
                    'Technology': 'XLK',
                    'Healthcare': 'XLV',
                    'Financials': 'XLF',
                    'Consumer Discretionary': 'XLY',
                    'Consumer Staples': 'XLP',
                    'Industrials': 'XLI',
                    'Energy': 'XLE',
                    'Utilities': 'XLU',
                    'Materials': 'XLB',
                    'Real Estate': 'XLRE',
                    'Communication Services': 'XLC'
                }
            elif market == 'Europe':
                sector_etfs = {
                    'Technology': 'SX8P.DE',  # STOXX Europe 600 Technology
                    'Healthcare': 'SXDP.DE',  # STOXX Europe 600 Healthcare
                    'Financials': 'SXFP.DE',  # STOXX Europe 600 Financials
                    'Consumer Products': 'SXQP.DE',  # STOXX Europe 600 Personal & Household Goods
                    'Food & Beverage': 'SX3P.DE',  # STOXX Europe 600 Food & Beverage
                    'Industrials': 'SXNP.DE',  # STOXX Europe 600 Industrials
                    'Energy': 'SXEP.DE',  # STOXX Europe 600 Oil & Gas
                    'Utilities': 'SX6P.DE',  # STOXX Europe 600 Utilities
                    'Basic Resources': 'SXPP.DE',  # STOXX Europe 600 Basic Resources
                    'Real Estate': 'SX86P.DE',  # STOXX Europe 600 Real Estate
                    'Telecom': 'SXKP.DE'  # STOXX Europe 600 Telecommunications
                }
            else:  # Global/Default
                sector_etfs = {
                    'Technology': 'IXN',  # iShares Global Tech ETF
                    'Healthcare': 'IXJ',  # iShares Global Healthcare ETF
                    'Financials': 'IXG',  # iShares Global Financials ETF
                    'Consumer Discretionary': 'RXI',  # iShares Global Consumer Discretionary ETF
                    'Consumer Staples': 'KXI',  # iShares Global Consumer Staples ETF
                    'Industrials': 'EXI',  # iShares Global Industrials ETF
                    'Energy': 'IXC',  # iShares Global Energy ETF
                    'Utilities': 'JXI',  # iShares Global Utilities ETF
                    'Materials': 'MXI',  # iShares Global Materials ETF
                    'Real Estate': 'REET',  # iShares Global REIT ETF
                    'Communication': 'IXP'  # iShares Global Comm Services ETF
                }
            
            # Convert period to yfinance format
            yf_period_map = {
                '1m': '1mo',
                '3m': '3mo',
                '6m': '6mo',
                '1y': '1y',
                '3y': '3y',
                '5y': '5y'
            }
            yf_period = yf_period_map.get(period, '6mo')
            
            # Get benchmark ETF based on market
            benchmark_etf = 'SPY'  # Default to S&P 500
            if market == 'Europe':
                benchmark_etf = 'FEZ'  # EURO STOXX 50 ETF
            elif market == 'Global':
                benchmark_etf = 'ACWI'  # iShares MSCI ACWI ETF
            
            # Fetch price data for sectors and benchmark
            price_data = {}
            
            # First get benchmark
            try:
                benchmark = yf.Ticker(benchmark_etf)
                benchmark_history = benchmark.history(period=yf_period)
                if not benchmark_history.empty:
                    price_data['Benchmark'] = benchmark_history['Close'].copy()
            except Exception as e:
                logger.warning(f"Error fetching benchmark data: {str(e)}")
            
            # Then get sector ETFs
            for sector_name, ticker in sector_etfs.items():
                try:
                    etf = yf.Ticker(ticker)
                    history = etf.history(period=yf_period)
                    
                    if not history.empty:
                        price_data[sector_name] = history['Close'].copy()
                    else:
                        logger.warning(f"No data retrieved for {sector_name} ({ticker})")
                except Exception as e:
                    logger.warning(f"Error fetching data for {sector_name} ({ticker}): {str(e)}")
            
            # Convert price data to DataFrame and calculate returns
            prices_df = pd.DataFrame(price_data)
            
            # Normalize prices to 100 at the start
            normalized_prices = prices_df.div(prices_df.iloc[0]) * 100
            
            # Calculate total returns for the period
            total_returns = {}
            for column in normalized_prices.columns:
                start_value = normalized_prices[column].iloc[0]
                end_value = normalized_prices[column].iloc[-1]
                total_return = ((end_value / start_value) - 1) * 100
                total_returns[column] = round(total_return, 2)
            
            # Calculate relative returns (vs benchmark)
            relative_returns = {}
            if 'Benchmark' in total_returns:
                benchmark_return = total_returns['Benchmark']
                for sector, sector_return in total_returns.items():
                    if sector != 'Benchmark':
                        relative_returns[sector] = round(sector_return - benchmark_return, 2)
            
            # Sort sectors by performance
            sorted_sectors = sorted(
                [(sector, perf) for sector, perf in total_returns.items() if sector != 'Benchmark'],
                key=lambda x: x[1],
                reverse=True
            )
            
            top_sectors = [sector for sector, _ in sorted_sectors[:3]]
            bottom_sectors = [sector for sector, _ in sorted_sectors[-3:]]
            
            # Identify current phase of business cycle based on sector leadership
            cycle_phases = {
                'Early Cycle': ['Consumer Discretionary', 'Financials', 'Industrials', 'Materials', 'Real Estate'],
                'Mid Cycle': ['Technology', 'Industrials', 'Energy', 'Materials'],
                'Late Cycle': ['Energy', 'Healthcare', 'Consumer Staples', 'Utilities', 'Materials'],
                'Recession': ['Healthcare', 'Consumer Staples', 'Utilities', 'Communication Services']
            }
            
            # Count how many top sectors belong to each phase
            phase_scores = {phase: 0 for phase in cycle_phases.keys()}
            for sector in top_sectors:
                for phase, sectors in cycle_phases.items():
                    if sector in sectors:
                        phase_scores[phase] += 1
            
            # Determine likely cycle phase
            likely_phase = max(phase_scores.items(), key=lambda x: x[1])[0]
            confidence = 'moderate'  # Default
            
            # If there's a tie or low confidence
            max_score = max(phase_scores.values())
            if list(phase_scores.values()).count(max_score) > 1:
                confidence = 'low'
                # Find tied phases
                tied_phases = [phase for phase, score in phase_scores.items() if score == max_score]
                likely_phase = ' or '.join(tied_phases)
            elif max_score >= 2:
                confidence = 'high'
            
            # Identify momentum shifts (sectors showing recent improvement)
            # For this, we'd need more granular data - this is a simplified approach
            # In a real implementation, you'd compare recent performance to longer-term performance
            
            momentum_shifts = []
            
            # For simplified demonstration - identify sectors with positive momentum
            # This would normally use technical analysis indicators or relative momentum metrics
            
            # Prepare result
            result = {
                "status": "success",
                "market": market,
                "analysis_period": period,
                "as_of_date": normalized_prices.index[-1].strftime('%Y-%m-%d'),
                "sector_performance": {
                    "total_returns_pct": total_returns,
                    "relative_returns_pct": relative_returns,
                    "leaders": top_sectors,
                    "laggards": bottom_sectors
                },
                "business_cycle": {
                    "likely_phase": likely_phase,
                    "confidence": confidence,
                    "supporting_evidence": f"Performance leadership in {', '.join(top_sectors)}",
                    "phase_descriptions": {
                        "Early Cycle": "Economic recovery, accelerating growth, accommodative policy",
                        "Mid Cycle": "Peak growth rates, tightening policy, strong earnings",
                        "Late Cycle": "Slowing growth, restrictive policy, margin pressures",
                        "Recession": "Negative growth, easing policy, defensive positioning"
                    }
                },
                "momentum_shifts": momentum_shifts,
                "data_source": "yfinance - sector ETF returns"
            }
            
            # Cache the result
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_duration
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing sector rotation: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to analyze sector rotation: {str(e)}"
            }
    
    def get_global_liquidity_analysis(self, include_central_banks=True):
        """
        Analyze global liquidity conditions and central bank balance sheets
        
        Args:
            include_central_banks (bool): Whether to include individual central bank data
            
        Returns:
            dict: Liquidity analysis
        """
        try:
            # Check cache
            cache_key = f"global_liquidity_{include_central_banks}"
            if cache_key in self.cache and datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
                return self.cache[cache_key]
            
            # In a real implementation, this would use actual central bank balance sheet data
            # For demonstration, we'll simulate the data
            
            analysis_date = datetime.now().strftime('%Y-%m-%d')
            
            # Simulated global liquidity trend data
            liquidity_trend = {
                "current_direction": "contracting",
                "rate_of_change": "moderate",
                "6m_change_pct": -2.5,
                "12m_change_pct": -5.8,
                "contributing_factors": [
                    "Coordinated quantitative tightening by major central banks",
                    "Rising interest rates slowing credit creation",
                    "Reduced fiscal stimulus compared to prior periods"
                ]
            }
            
            # Simulated central bank balance sheet data
            central_banks = {
                "Federal Reserve": {
                    "current_balance_sheet_usd_trillion": 7.4,
                    "6m_change_pct": -3.2,
                    "12m_change_pct": -8.5,
                    "trend": "contracting",
                    "policy_stance": "quantitative tightening"
                },
                "European Central Bank": {
                    "current_balance_sheet_usd_trillion": 8.2,
                    "6m_change_pct": -2.1,
                    "12m_change_pct": -4.7,
                    "trend": "contracting",
                    "policy_stance": "quantitative tightening"
                },
                "Bank of Japan": {
                    "current_balance_sheet_usd_trillion": 5.3,
                    "6m_change_pct": 0.8,
                    "12m_change_pct": 2.1,
                    "trend": "expanding",
                    "policy_stance": "accommodative"
                },
                "People's Bank of China": {
                    "current_balance_sheet_usd_trillion": 5.8,
                    "6m_change_pct": 1.2,
                    "12m_change_pct": 3.5,
                    "trend": "expanding",
                    "policy_stance": "accommodative"
                }
            }
            
            # Calculate aggregate data
            total_balance_sheet = sum(cb['current_balance_sheet_usd_trillion'] for cb in central_banks.values())
            
            # Weighted average changes
            weighted_6m_change = sum(cb['current_balance_sheet_usd_trillion'] * cb['6m_change_pct'] for cb in central_banks.values()) / total_balance_sheet
            weighted_12m_change = sum(cb['current_balance_sheet_usd_trillion'] * cb['12m_change_pct'] for cb in central_banks.values()) / total_balance_sheet
            
            # Market implications based on liquidity trends
            if liquidity_trend['current_direction'] == "contracting" and liquidity_trend['rate_of_change'] in ["moderate", "rapid"]:
                liquidity_implications = {
                    "overall_impact": "Restrictive",
                    "asset_class_impacts": {
                        "equities": "Negative - valuation pressure, especially for growth stocks",
                        "bonds": "Negative - upward pressure on yields, particularly longer-duration assets",
                        "credit": "Negative - widening spreads, especially in high yield",
                        "real_assets": "Mixed - inflation hedge benefits offset by higher discount rates"
                    },
                    "key_risks": [
                        "Market stress episodes",
                        "Credit market liquidity deterioration",
                        "Asset price volatility"
                    ]
                }
            elif liquidity_trend['current_direction'] == "contracting" and liquidity_trend['rate_of_change'] == "slow":
                liquidity_implications = {
                    "overall_impact": "Mildly Restrictive",
                    "asset_class_impacts": {
                        "equities": "Mildly negative - gradual valuation adjustment",
                        "bonds": "Mildly negative - modest upward pressure on yields",
                        "credit": "Neutral to negative - selective pressure on lower quality issuers",
                        "real_assets": "Mixed - depends on inflation trajectory"
                    },
                    "key_risks": [
                        "Increased volatility during policy transitions",
                        "Sector rotation"
                    ]
                }
            elif liquidity_trend['current_direction'] == "expanding":
                liquidity_implications = {
                    "overall_impact": "Supportive",
                    "asset_class_impacts": {
                        "equities": "Positive - multiple expansion potential",
                        "bonds": "Mixed - support for prices but potential inflation concerns",
                        "credit": "Positive - spread compression likely",
                        "real_assets": "Positive - particularly with inflation concerns"
                    },
                    "key_risks": [
                        "Asset price bubbles",
                        "Inflation acceleration",
                        "Currency debasement concerns"
                    ]
                }
            else:
                liquidity_implications = {
                    "overall_impact": "Neutral",
                    "asset_class_impacts": {
                        "equities": "Neutral - focus on fundamentals and earnings",
                        "bonds": "Neutral - range-bound yields likely",
                        "credit": "Neutral - focus on credit selection",
                        "real_assets": "Neutral - dependent on other factors"
                    },
                    "key_risks": [
                        "Policy shifts",
                        "Growth/inflation dynamics"
                    ]
                }
            
            # Prepare result
            result = {
                "status": "success",
                "analysis_date": analysis_date,
                "global_liquidity_assessment": {
                    "trend": liquidity_trend,
                    "total_major_cb_balance_sheets_usd_trillion": round(total_balance_sheet, 1),
                    "aggregate_6m_change_pct": round(weighted_6m_change, 1),
                    "aggregate_12m_change_pct": round(weighted_12m_change, 1)
                },
                "market_implications": liquidity_implications,
                "data_source": "Simulated data - would use central bank data in production"
            }
            
            # Include central bank details if requested
            if include_central_banks:
                result["central_banks"] = central_banks
            
            # Cache the result
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_duration
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing global liquidity: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to analyze global liquidity: {str(e)}"
            }