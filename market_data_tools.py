import os
import logging
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketDataTools:
    """Implementation of tools for market data analysis agent"""
    
    def __init__(self, api_keys=None):
        """Initialize with necessary API keys"""
        self.api_keys = api_keys or {}
        self.alpha_vantage_key = self.api_keys.get('alpha_vantage', '')
        self.polygon_key = self.api_keys.get('polygon', '')
        
        # Cache for expensive API calls
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 15 * 60  # 15 minutes in seconds - shorter for market data
    
    def get_historical_prices(self, symbol, interval="1d", period="1mo"):
        """
        Fetch historical price data for a given symbol
        
        Args:
            symbol (str): The ticker symbol to fetch data for
            interval (str): Time interval between data points (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
            period (str): The time period to fetch data for (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            dict: Historical price data with OHLCV and derived metrics
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{interval}_{period}"
            if cache_key in self.cache and datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
                return self.cache[cache_key]
            
            logger.info(f"Fetching historical data for {symbol} (interval: {interval}, period: {period})")
            
            # Try to get data from yfinance
            ticker = yf.Ticker(symbol)
            history = ticker.history(period=period, interval=interval)
            
            if history.empty:
                return {
                    "status": "error",
                    "message": f"No data available for {symbol}"
                }
            
            # Process the data
            ohlcv_data = {}
            for date, row in history.iterrows():
                date_str = date.strftime('%Y-%m-%d') if interval in ['1d', '1wk', '1mo'] else date.strftime('%Y-%m-%d %H:%M:%S')
                ohlcv_data[date_str] = {
                    "open": round(row['Open'], 2),
                    "high": round(row['High'], 2),
                    "low": round(row['Low'], 2),
                    "close": round(row['Close'], 2),
                    "volume": int(row['Volume'])
                }
            
            # Calculate basic stats
            latest_close = history['Close'].iloc[-1]
            previous_close = history['Close'].iloc[-2] if len(history) > 1 else None
            
            high_52wk = history['High'].max()
            low_52wk = history['Low'].min()
            
            change = latest_close - previous_close if previous_close else None
            change_pct = (change / previous_close * 100) if change is not None and previous_close else None
            
            avg_volume = history['Volume'].mean()
            latest_volume = history['Volume'].iloc[-1]
            volume_ratio = latest_volume / avg_volume if avg_volume else None
            
            # Summary stats
            summary = {
                "latest_close": round(latest_close, 2),
                "previous_close": round(previous_close, 2) if previous_close else None,
                "change": round(change, 2) if change is not None else None,
                "change_pct": round(change_pct, 2) if change_pct is not None else None,
                "high_52wk": round(high_52wk, 2),
                "low_52wk": round(low_52wk, 2),
                "avg_volume": int(avg_volume),
                "latest_volume": int(latest_volume),
                "volume_ratio": round(volume_ratio, 2) if volume_ratio else None
            }
            
            result = {
                "status": "success",
                "symbol": symbol,
                "interval": interval,
                "period": period,
                "data_points": len(ohlcv_data),
                "summary": summary,
                "ohlcv_data": ohlcv_data,
                "data_source": "Yahoo Finance"
            }
            
            # Cache the result
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_duration
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to fetch historical data: {str(e)}"
            }
    
    def calculate_technical_indicators(self, symbol, indicators=None, interval="1d", period="6mo"):
        """
        Calculate technical indicators for a given symbol
        
        Args:
            symbol (str): The ticker symbol to analyze
            indicators (list): List of indicators to calculate (e.g., ['sma', 'rsi', 'macd'])
            interval (str): Time interval between data points
            period (str): The time period to analyze
            
        Returns:
            dict: Technical indicators with values and signals
        """
        try:
            # Default indicators if none specified
            if not indicators:
                indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 'adx']
            
            # Normalize indicator names
            indicators = [ind.lower() for ind in indicators]
            
            # Check cache
            cache_key = f"{symbol}_indicators_{interval}_{period}_{'-'.join(indicators)}"
            if cache_key in self.cache and datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
                return self.cache[cache_key]
            
            logger.info(f"Calculating technical indicators for {symbol} ({', '.join(indicators)})")
            
            # Get historical data
            try:
                # Fetch data
                ticker = yf.Ticker(symbol)
                history = ticker.history(period=period, interval=interval)
                
                if history.empty:
                    return {
                        "status": "error",
                        "message": f"No historical data available for {symbol}"
                    }
                
                # Extract price and volume data
                close = history['Close'].values
                high = history['High'].values
                low = history['Low'].values
                open_prices = history['Open'].values
                volume = history['Volume'].values
                
                # Prepare result structure
                indicator_results = {}
                signals = {}
                
                # Calculate indicators
                # SMA - Simple Moving Average
                if 'sma' in indicators:
                    sma_short = self._calculate_sma(close, 20)
                    sma_medium = self._calculate_sma(close, 50)
                    sma_long = self._calculate_sma(close, 200)
                    
                    indicator_results['sma'] = {
                        'sma20': self._format_indicator_series(history.index, sma_short),
                        'sma50': self._format_indicator_series(history.index, sma_medium),
                        'sma200': self._format_indicator_series(history.index, sma_long)
                    }
                    
                    # SMA signals
                    if len(sma_short) > 1 and len(sma_medium) > 1:
                        if sma_short[-1] > sma_medium[-1] and sma_short[-2] <= sma_medium[-2]:
                            signals['sma_crossover'] = "Bullish crossover (SMA20 crossed above SMA50)"
                        elif sma_short[-1] < sma_medium[-1] and sma_short[-2] >= sma_medium[-2]:
                            signals['sma_crossover'] = "Bearish crossover (SMA20 crossed below SMA50)"
                    
                    if len(sma_long) > 0:
                        if close[-1] > sma_long[-1]:
                            signals['trend_sma200'] = "Bullish (Price above SMA200)"
                        else:
                            signals['trend_sma200'] = "Bearish (Price below SMA200)"
                
                # EMA - Exponential Moving Average
                if 'ema' in indicators:
                    ema_short = self._calculate_ema(close, 12)
                    ema_medium = self._calculate_ema(close, 26)
                    ema_long = self._calculate_ema(close, 50)
                    
                    indicator_results['ema'] = {
                        'ema12': self._format_indicator_series(history.index, ema_short),
                        'ema26': self._format_indicator_series(history.index, ema_medium),
                        'ema50': self._format_indicator_series(history.index, ema_long)
                    }
                    
                    # EMA signals
                    if len(ema_short) > 1 and len(ema_medium) > 1:
                        if ema_short[-1] > ema_medium[-1] and ema_short[-2] <= ema_medium[-2]:
                            signals['ema_crossover'] = "Bullish crossover (EMA12 crossed above EMA26)"
                        elif ema_short[-1] < ema_medium[-1] and ema_short[-2] >= ema_medium[-2]:
                            signals['ema_crossover'] = "Bearish crossover (EMA12 crossed below EMA26)"
                
                # RSI - Relative Strength Index
                if 'rsi' in indicators:
                    rsi_values = self._calculate_rsi(close, 14)
                    
                    indicator_results['rsi'] = {
                        'rsi14': self._format_indicator_series(history.index, rsi_values)
                    }
                    
                    # RSI signals
                    latest_rsi = rsi_values[-1]
                    
                    if latest_rsi < 30:
                        signals['rsi'] = "Oversold (RSI < 30)"
                    elif latest_rsi > 70:
                        signals['rsi'] = "Overbought (RSI > 70)"
                    else:
                        signals['rsi'] = "Neutral"
                
                # MACD - Moving Average Convergence Divergence
                if 'macd' in indicators:
                    macd_line, signal_line, histogram = self._calculate_macd(close)
                    
                    indicator_results['macd'] = {
                        'macd_line': self._format_indicator_series(history.index, macd_line),
                        'signal_line': self._format_indicator_series(history.index, signal_line),
                        'histogram': self._format_indicator_series(history.index, histogram)
                    }
                    
                    # MACD signals
                    if len(macd_line) > 1 and len(signal_line) > 1:
                        if macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2]:
                            signals['macd'] = "Bullish crossover (MACD crossed above signal line)"
                        elif macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2]:
                            signals['macd'] = "Bearish crossover (MACD crossed below signal line)"
                        elif macd_line[-1] > signal_line[-1]:
                            signals['macd'] = "Bullish (MACD above signal line)"
                        else:
                            signals['macd'] = "Bearish (MACD below signal line)"
                
                # Bollinger Bands
                if 'bollinger' in indicators:
                    upper_band, middle_band, lower_band = self._calculate_bollinger_bands(close)
                    
                    indicator_results['bollinger_bands'] = {
                        'upper_band': self._format_indicator_series(history.index, upper_band),
                        'middle_band': self._format_indicator_series(history.index, middle_band),
                        'lower_band': self._format_indicator_series(history.index, lower_band)
                    }
                    
                    # Bollinger Band signals
                    latest_close = close[-1]
                    latest_upper = upper_band[-1]
                    latest_lower = lower_band[-1]
                    
                    if not np.isnan(latest_upper) and not np.isnan(latest_lower):
                        bandwidth = (latest_upper - latest_lower) / latest_close
                        
                        if latest_close > latest_upper:
                            signals['bollinger'] = "Overbought (Price above upper band)"
                        elif latest_close < latest_lower:
                            signals['bollinger'] = "Oversold (Price below lower band)"
                        else:
                            signals['bollinger'] = "Within bands"
                        
                        if bandwidth < 0.1:  # Arbitrary threshold for tight bands
                            signals['bollinger_bandwidth'] = "Tight bands (potential breakout)"
                
                # ATR - Average True Range (Volatility)
                if 'atr' in indicators:
                    atr_values = self._calculate_atr(high, low, close, 14)
                    
                    indicator_results['atr'] = {
                        'atr14': self._format_indicator_series(history.index, atr_values)
                    }
                    
                    # ATR signals (purely informative, not directional)
                    if len(atr_values) >= 30:
                        avg_atr = np.nanmean(atr_values[-30:])  # Average of last 30 periods
                        latest_atr = atr_values[-1]
                        
                        if latest_atr > avg_atr * 1.5:
                            signals['volatility'] = "High volatility (ATR above average)"
                        elif latest_atr < avg_atr * 0.5:
                            signals['volatility'] = "Low volatility (ATR below average)"
                        else:
                            signals['volatility'] = "Normal volatility"
                
                # Determine overall technical signal
                bullish_signals = sum(1 for signal in signals.values() if 'Bullish' in signal or 'Oversold' in signal)
                bearish_signals = sum(1 for signal in signals.values() if 'Bearish' in signal or 'Overbought' in signal)
                
                if bullish_signals > bearish_signals + 1:
                    overall_signal = "Bullish"
                elif bearish_signals > bullish_signals + 1:
                    overall_signal = "Bearish"
                else:
                    overall_signal = "Neutral"
                
                result = {
                    "status": "success",
                    "symbol": symbol,
                    "as_of_date": history.index[-1].strftime('%Y-%m-%d'),
                    "indicators": indicator_results,
                    "signals": signals,
                    "overall_signal": overall_signal
                }
                
                # Cache the result
                self.cache[cache_key] = result
                self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_duration
                
                return result
                
            except Exception as e:
                logger.error(f"Error calculating indicators: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Error calculating indicators: {str(e)}"
                }
            
        except Exception as e:
            logger.error(f"Error in technical analysis for {symbol}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to perform technical analysis: {str(e)}"
            }
    
    def _format_indicator_series(self, dates, values):
        """Format indicator series into a date-keyed dictionary"""
        result = {}
        for i, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')
            if i < len(values) and not np.isnan(values[i]):
                result[date_str] = round(float(values[i]), 4)
        return result
    
    def _calculate_sma(self, data, window):
        """Calculate Simple Moving Average"""
        return pd.Series(data).rolling(window=window).mean().values
    
    def _calculate_ema(self, data, span):
        """Calculate Exponential Moving Average"""
        return pd.Series(data).ewm(span=span, adjust=False).mean().values
    
    def _calculate_rsi(self, data, window):
        """Calculate Relative Strength Index"""
        # Calculate price changes
        delta = pd.Series(data).diff().dropna().values
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gain = pd.Series(gains).rolling(window=window).mean().values
        avg_loss = pd.Series(losses).rolling(window=window).mean().values
        
        # Calculate RS and RSI
        rs = np.zeros_like(avg_gain)
        rsi = np.zeros_like(avg_gain)
        
        for i in range(len(avg_gain)):
            if avg_loss[i] == 0:
                rs[i] = 100.0
            else:
                rs[i] = avg_gain[i] / avg_loss[i]
            
            rsi[i] = 100 - (100 / (1 + rs[i]))
        
        # Pad the beginning to match the original data length
        pad_length = len(data) - len(rsi)
        padded_rsi = np.concatenate([np.array([np.nan] * pad_length), rsi])
        
        return padded_rsi
    
    def _calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        """Calculate MACD, signal line, and histogram"""
        # Calculate EMAs
        ema_fast = self._calculate_ema(data, fast_period)
        ema_slow = self._calculate_ema(data, slow_period)
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = pd.Series(macd_line).ewm(span=signal_period, adjust=False).mean().values
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, data, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        # Calculate SMA (middle band)
        middle_band = pd.Series(data).rolling(window=window).mean().values
        
        # Calculate standard deviation
        std = pd.Series(data).rolling(window=window).std().values
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return upper_band, middle_band, lower_band
    
    def _calculate_atr(self, high, low, close, window=14):
        """Calculate Average True Range"""
        # Create shifted close
        prev_close = np.concatenate([[close[0]], close[:-1]])
        
        # Calculate true range
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        # Get the maximum of the three
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # Calculate ATR as EMA of true range
        atr = pd.Series(tr).ewm(span=window, adjust=False).mean().values
        
        return atr
    
    def analyze_market_structure(self, symbol, interval="1d", period="1y"):
        """
        Analyze market structure including support/resistance, trends, and patterns
        
        Args:
            symbol (str): The ticker symbol to analyze
            interval (str): Time interval between data points
            period (str): The time period to analyze
            
        Returns:
            dict: Market structure analysis including support/resistance levels
        """
        try:
            # Check cache
            cache_key = f"{symbol}_structure_{interval}_{period}"
            if cache_key in self.cache and datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
                return self.cache[cache_key]
            
            logger.info(f"Analyzing market structure for {symbol}")
            
            # Get historical data
            ticker = yf.Ticker(symbol)
            history = ticker.history(period=period, interval=interval)
            
            if history.empty:
                return {
                    "status": "error",
                    "message": f"No historical data available for {symbol}"
                }
            
            # Extract price data
            close = history['Close'].values
            high = history['High'].values
            low = history['Low'].values
            
            # Find significant pivot highs and lows (simplified algorithm)
            window = 5  # Look 5 candles before and after
            
            pivot_highs = []
            pivot_lows = []
            
            for i in range(window, len(high) - window):
                # Check if this is a pivot high
                if high[i] == max(high[i-window:i+window+1]):
                    pivot_highs.append((history.index[i].strftime('%Y-%m-%d'), high[i]))
                
                # Check if this is a pivot low
                if low[i] == min(low[i-window:i+window+1]):
                    pivot_lows.append((history.index[i].strftime('%Y-%m-%d'), low[i]))
            
            # Filter nearby pivots (within 2% of price)
            filtered_highs = self._filter_nearby_levels(pivot_highs, 0.02)
            filtered_lows = self._filter_nearby_levels(pivot_lows, 0.02)
            
            # Find horizontal support and resistance levels based on pivot frequency
            support_levels = self._identify_key_levels(filtered_lows, close[-1])
            resistance_levels = self._identify_key_levels(filtered_highs, close[-1])
            
            # Identify trend direction
            sma_short = self._calculate_sma(close, 20)[-1] if len(close) >= 20 else None
            sma_long = self._calculate_sma(close, 50)[-1] if len(close) >= 50 else None
            
            if sma_short is not None and sma_long is not None:
                if close[-1] > sma_short > sma_long:
                    trend = "Uptrend"
                elif close[-1] < sma_short < sma_long:
                    trend = "Downtrend"
                else:
                    trend = "Sideways/Neutral"
            else:
                # If we don't have enough data for both SMAs, use a simpler method
                if len(close) >= 10:
                    first_half_avg = np.mean(close[:len(close)//2])
                    second_half_avg = np.mean(close[len(close)//2:])
                    
                    if second_half_avg > first_half_avg * 1.03:  # 3% higher
                        trend = "Uptrend"
                    elif second_half_avg < first_half_avg * 0.97:  # 3% lower
                        trend = "Downtrend"
                    else:
                        trend = "Sideways/Neutral"
                else:
                    trend = "Insufficient data"
            
            # Check for common chart patterns (very simplified)
            patterns = []
            if len(close) >= 20:
                # Simple pattern detection based on price behavior
                if trend == "Uptrend" and len(pivot_lows) >= 2:
                    # Higher lows pattern in uptrend
                    recent_lows = sorted(pivot_lows, key=lambda x: x[0], reverse=True)[:2]
                    if recent_lows[0][1] > recent_lows[1][1]:
                        patterns.append("Higher Lows Pattern (Bullish)")
                
                if trend == "Downtrend" and len(pivot_highs) >= 2:
                    # Lower highs pattern in downtrend
                    recent_highs = sorted(pivot_highs, key=lambda x: x[0], reverse=True)[:2]
                    if recent_highs[0][1] < recent_highs[1][1]:
                        patterns.append("Lower Highs Pattern (Bearish)")
                
                # Check for potential double bottom (very simplified)
                if len(pivot_lows) >= 2:
                    recent_lows = sorted(pivot_lows, key=lambda x: x[0], reverse=True)[:2]
                    price_diff = abs(recent_lows[0][1] - recent_lows[1][1])
                    relative_diff = price_diff / recent_lows[0][1]
                    
                    if relative_diff < 0.03:  # Within 3%
                        patterns.append("Potential Double Bottom")
                
                # Check for potential double top
                if len(pivot_highs) >= 2:
                    recent_highs = sorted(pivot_highs, key=lambda x: x[0], reverse=True)[:2]
                    price_diff = abs(recent_highs[0][1] - recent_highs[1][1])
                    relative_diff = price_diff / recent_highs[0][1]
                    
                    if relative_diff < 0.03:  # Within 3%
                        patterns.append("Potential Double Top")
            
            # Determine key areas to watch
            latest_close = close[-1]
            
            key_support = None
            for level in sorted([float(k) for k in support_levels.keys()]):
                if level < latest_close:
                    key_support = level
                    break
            
            key_resistance = None
            for level in sorted([float(k) for k in resistance_levels.keys()], reverse=True):
                if level > latest_close:
                    key_resistance = level
                    break
            
            result = {
                "status": "success",
                "symbol": symbol,
                "as_of_date": history.index[-1].strftime('%Y-%m-%d'),
                "current_price": round(latest_close, 2),
                "market_structure": {
                    "trend": trend,
                    "support_levels": {round(float(k), 2): v for k, v in support_levels.items()},
                    "resistance_levels": {round(float(k), 2): v for k, v in resistance_levels.items()},
                    "key_support": round(float(key_support), 2) if key_support else None,
                    "key_resistance": round(float(key_resistance), 2) if key_resistance else None,
                    "patterns": patterns
                },
                "technical_outlook": self._generate_technical_outlook(trend, patterns, latest_close, key_support, key_resistance)
            }
            
            # Cache the result
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_duration
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing market structure for {symbol}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to analyze market structure: {str(e)}"
            }
    
    def _filter_nearby_levels(self, levels, threshold):
        """Filter out levels that are very close to each other"""
        if not levels:
            return []
        
        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x[1])
        
        filtered = [sorted_levels[0]]
        
        for date, price in sorted_levels[1:]:
            # Check if this price is significantly different from the last accepted price
            last_price = filtered[-1][1]
            
            # Calculate relative difference
            rel_diff = abs(price - last_price) / last_price
            
            if rel_diff > threshold:
                filtered.append((date, price))
        
        return filtered
    
    def _identify_key_levels(self, pivots, current_price):
        """Identify key support/resistance levels based on pivot frequency"""
        if not pivots:
            return {}
        
        # Group nearby levels and count occurrences
        level_counts = {}
        
        for date, price in pivots:
            # Find closest existing level
            closest_level = None
            min_diff = float('inf')
            
            for level in level_counts.keys():
                diff = abs(price - level)
                rel_diff = diff / price
                
                if rel_diff < 0.02 and rel_diff < min_diff:  # Within 2%
                    closest_level = level
                    min_diff = rel_diff
            
            if closest_level is not None:
                # Add to existing level
                level_counts[closest_level]["count"] += 1
                level_counts[closest_level]["dates"].append(date)
            else:
                # Create new level
                level_counts[price] = {
                    "count": 1,
                    "dates": [date]
                }
        
        # Sort by relevance (count and proximity to current price)
        sorted_levels = {}
        for price, data in level_counts.items():
            # Calculate relevance score based on count and recency
            rel_diff = abs(price - current_price) / current_price
            proximity_score = 1 / (1 + rel_diff)  # Higher for closer levels
            
            # Factor in recency of tests
            dates_parsed = [datetime.strptime(d, '%Y-%m-%d') for d in data["dates"]]
            most_recent = max(dates_parsed)
            days_since = (datetime.now() - most_recent).days
            recency_score = 1 / (1 + days_since/30)  # Higher for more recent tests
            
            relevance = data["count"] * proximity_score * recency_score
            
            sorted_levels[price] = {
                "count": data["count"],
                "relevance": round(relevance, 2),
                "last_test": most_recent.strftime('%Y-%m-%d')
            }
        
        # Return top 5 most relevant levels
        return dict(sorted(sorted_levels.items(), key=lambda x: x[1]["relevance"], reverse=True)[:5])
    
    def _generate_technical_outlook(self, trend, patterns, current_price, support, resistance):
       """Generate a technical outlook based on identified patterns and levels"""
       outlook = []
       
       # Add trend assessment
       if trend == "Uptrend":
           outlook.append("The price is in an uptrend, trading above key moving averages.")
       elif trend == "Downtrend":
           outlook.append("The price is in a downtrend, trading below key moving averages.")
       else:
           outlook.append("The price is trading sideways, indicating market indecision.")
       
       # Add pattern implications
       bullish_patterns = ["Potential Double Bottom", "Higher Lows Pattern (Bullish)"]
       bearish_patterns = ["Potential Double Top", "Lower Highs Pattern (Bearish)"]
       
       bullish_count = sum(1 for p in patterns if p in bullish_patterns)
       bearish_count = sum(1 for p in patterns if p in bearish_patterns)
       
       if patterns:
           pattern_text = "Notable chart patterns: " + ", ".join(patterns)
           outlook.append(pattern_text)
       
       # Add support/resistance assessment
       if support is not None:
           support_distance = (current_price - support) / current_price * 100
           outlook.append(f"Nearest support at {round(support, 2)} ({round(support_distance, 1)}% below current price).")
       
       if resistance is not None:
           resistance_distance = (resistance - current_price) / current_price * 100
           outlook.append(f"Nearest resistance at {round(resistance, 2)} ({round(resistance_distance, 1)}% above current price).")
       
       # Add overall bias
       if bullish_count > bearish_count and trend != "Downtrend":
           outlook.append("Technical bias: Bullish")
       elif bearish_count > bullish_count and trend != "Uptrend":
           outlook.append("Technical bias: Bearish")
       else:
           outlook.append("Technical bias: Neutral")
       
       return outlook
   
    def analyze_volume_profile(self, symbol, interval="1d", period="3mo"):
        """
        Analyze volume profile and volume-based indicators
        
        Args:
            symbol (str): The ticker symbol to analyze
            interval (str): Time interval between data points
            period (str): The time period to analyze
            
        Returns:
            dict: Volume analysis including volume profile and indicators
        """
        try:
            # Check cache
            cache_key = f"{symbol}_volume_{interval}_{period}"
            if cache_key in self.cache and datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
                return self.cache[cache_key]
            
            logger.info(f"Analyzing volume profile for {symbol}")
            
            # Get historical data
            ticker = yf.Ticker(symbol)
            history = ticker.history(period=period, interval=interval)
            
            if history.empty:
                return {
                    "status": "error",
                    "message": f"No historical data available for {symbol}"
                }
            
            # Calculate volume-based indicators
            volume = history['Volume'].values
            close = history['Close'].values
            
            # On-Balance Volume (OBV)
            obv = np.zeros_like(volume)
            obv[0] = volume[0]
            
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv[i] = obv[i-1] + volume[i]
                elif close[i] < close[i-1]:
                    obv[i] = obv[i-1] - volume[i]
                else:
                    obv[i] = obv[i-1]
            
            # Volume SMA
            volume_sma = pd.Series(volume).rolling(window=20).mean().values
            
            # Relative Volume (compared to 20-day average)
            rel_volume = np.where(volume_sma != 0, volume / volume_sma, np.nan)
            
            # Chaikin Money Flow (CMF)
            money_flow_multiplier = ((close - history['Low']) - (history['High'] - close)) / (history['High'] - history['Low'])
            money_flow_volume = money_flow_multiplier * volume
            
            # Handle division by zero or very small numbers
            cmf_denominator = pd.Series(volume).rolling(window=20).sum()
            cmf = pd.Series(money_flow_volume).rolling(window=20).sum() / cmf_denominator
            
            # Volume Weighted Average Price (VWAP) - for intraday this would be daily, but we'll calculate period VWAP
            total_volume = np.sum(volume)
            if total_volume > 0:
                vwap = np.sum(close * volume) / total_volume
            else:
                vwap = np.nan
            
            # Create volume profile (price levels with most volume)
            # Divide price range into bins
            price_min = min(history['Low'])
            price_max = max(history['High'])
            
            n_bins = 10
            bin_size = (price_max - price_min) / n_bins
            
            volume_profile = {}
            for i in range(n_bins):
                bin_low = price_min + i * bin_size
                bin_high = price_min + (i + 1) * bin_size
                bin_mid = (bin_low + bin_high) / 2
                
                # Find volume in this price range
                mask = (history['Low'] <= bin_high) & (history['High'] >= bin_low)
                bin_volume = np.sum(history.loc[mask, 'Volume'])
                
                volume_profile[round(bin_mid, 2)] = int(bin_volume)
            
            # Order by volume (highest to lowest)
            volume_profile = dict(sorted(volume_profile.items(), key=lambda x: x[1], reverse=True))
            
            # Identify high volume nodes (value areas)
            value_area = list(volume_profile.keys())[:3]
            
            # Analyze volume trends
            recent_volume = volume[-5:]
            avg_recent_volume = np.mean(recent_volume)
            avg_overall_volume = np.mean(volume)
            
            volume_trend = "Increasing" if avg_recent_volume > avg_overall_volume * 1.2 else \
                        "Decreasing" if avg_recent_volume < avg_overall_volume * 0.8 else "Stable"
            
            # Generate volume signals
            signals = []
            
            # Price/volume divergence
            if len(close) >= 5 and len(obv) >= 5:
                if close[-1] > close[-5] and obv[-1] < obv[-5]:
                    signals.append("Bearish divergence: Price up, OBV down")
                elif close[-1] < close[-5] and obv[-1] > obv[-5]:
                    signals.append("Bullish divergence: Price down, OBV up")
            
            # Unusual volume
            if len(rel_volume) > 0 and not np.isnan(rel_volume[-1]):
                if rel_volume[-1] > 2:
                    signals.append(f"Unusually high volume: {round(rel_volume[-1], 1)}x average")
            
            # CMF signals
            if not cmf.empty and not np.isnan(cmf.iloc[-1]):
                if cmf.iloc[-1] > 0.1:
                    signals.append("Strong buying pressure (CMF > 0.1)")
                elif cmf.iloc[-1] < -0.1:
                    signals.append("Strong selling pressure (CMF < -0.1)")
            
            # VWAP relationship
            if not np.isnan(vwap):
                if close[-1] > vwap:
                    signals.append("Price above VWAP (bullish)")
                else:
                    signals.append("Price below VWAP (bearish)")
            
            result = {
                "status": "success",
                "symbol": symbol,
                "as_of_date": history.index[-1].strftime('%Y-%m-%d'),
                "volume_analysis": {
                    "current_volume": int(volume[-1]),
                    "average_volume_20d": int(volume_sma[-1]) if not np.isnan(volume_sma[-1]) else None,
                    "relative_volume": round(rel_volume[-1], 2) if len(rel_volume) > 0 and not np.isnan(rel_volume[-1]) else None,
                    "volume_trend": volume_trend,
                    "cmf": round(float(cmf.iloc[-1]), 3) if not cmf.empty and not np.isnan(cmf.iloc[-1]) else None,
                    "vwap": round(vwap, 2) if not np.isnan(vwap) else None,
                    "volume_signals": signals,
                    "high_volume_nodes": [round(float(price), 2) for price in value_area]
                },
                "volume_profile": volume_profile,
                "obv_trend": "Bullish" if len(obv) > 20 and obv[-1] > obv[-20] else "Bearish" if len(obv) > 20 and obv[-1] < obv[-20] else "Neutral"
            }
            
            # Cache the result
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_duration
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing volume profile for {symbol}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to analyze volume profile: {str(e)}"
            }
   
    def assess_liquidity_conditions(self, symbol, interval="1d", period="3mo"):
        """
        Assess market liquidity conditions including bid-ask spreads,
        market depth, and trading costs
        
        Args:
            symbol (str): The ticker symbol to analyze
            interval (str): Time interval between data points
            period (str): The time period to analyze
            
        Returns:
            dict: Liquidity assessment and metrics
        """
        try:
            # Check cache
            cache_key = f"{symbol}_liquidity_{interval}_{period}"
            if cache_key in self.cache and datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
                return self.cache[cache_key]
            
            logger.info(f"Assessing liquidity conditions for {symbol}")
            
            # Get historical data
            ticker = yf.Ticker(symbol)
            history = ticker.history(period=period, interval=interval)
            
            if history.empty:
                return {
                    "status": "error",
                    "message": f"No historical data available for {symbol}"
                }
            
            # For a proper liquidity assessment, we'd need bid-ask data, order book depth, etc.
            # However, with just OHLCV data, we'll use proxy metrics:
            
            # 1. Volume metrics
            avg_daily_volume = history['Volume'].mean()
            recent_volume = history['Volume'].iloc[-5:].mean()
            volume_trend = "Increasing" if recent_volume > avg_daily_volume * 1.1 else \
                            "Decreasing" if recent_volume < avg_daily_volume * 0.9 else "Stable"
            
            # 2. Volatility-adjusted volume
            avg_price = history['Close'].mean()
            dollar_volume = avg_daily_volume * avg_price
            
            # 3. Amihud illiquidity ratio (daily absolute return divided by dollar volume)
            returns = history['Close'].pct_change().abs()
            volumes = history['Volume'] * history['Close']
            amihud_illiquidity = (returns / volumes).mean() * 10**6  # Scale for readability
            
            # 4. High-Low spread as liquidity proxy
            hl_spread = (history['High'] - history['Low']) / history['Close']
            avg_hl_spread = hl_spread.mean() * 100  # Convert to percentage
            
            # 5. Estimated bid-ask spread (using Roll's model - very rough approximation)
            # Roll's model estimates spread from return autocovariance
            returns = history['Close'].pct_change().dropna()
            if len(returns) > 1:
                cov = returns.iloc[:-1].values * returns.iloc[1:].values
                roll_spread = 2 * np.sqrt(-np.mean(cov)) if np.mean(cov) < 0 else np.nan
                roll_spread_pct = roll_spread * 100 if not np.isnan(roll_spread) else None
            else:
                roll_spread_pct = None
            
            # Liquidity classification based on volume and estimates
            if dollar_volume > 50000000:  # $50M daily
                liquidity_classification = "High"
                tradability = "Excellent"
                impact_assessment = "Minimal market impact for most order sizes"
            elif dollar_volume > 10000000:  # $10M daily
                liquidity_classification = "Medium-High"
                tradability = "Good"
                impact_assessment = "Minimal impact for moderate order sizes"
            elif dollar_volume > 1000000:  # $1M daily
                liquidity_classification = "Medium"
                tradability = "Fair"
                impact_assessment = "May experience impact with larger orders"
            elif dollar_volume > 100000:  # $100K daily
                liquidity_classification = "Medium-Low"
                tradability = "Challenging"
                impact_assessment = "Likely to experience significant impact with larger orders"
            else:
                liquidity_classification = "Low"
                tradability = "Difficult"
                impact_assessment = "High market impact even with moderate order sizes"
            
            result = {
                "status": "success",
                "symbol": symbol,
                "as_of_date": history.index[-1].strftime('%Y-%m-%d'),
                "liquidity_metrics": {
                    "avg_daily_volume": int(avg_daily_volume),
                    "recent_volume_trend": volume_trend,
                    "dollar_volume": round(dollar_volume, 2),
                    "amihud_illiquidity": round(amihud_illiquidity, 6) if not np.isnan(amihud_illiquidity) else None,
                    "high_low_spread_pct": round(avg_hl_spread, 2),
                    "estimated_bid_ask_spread_pct": round(roll_spread_pct, 4) if roll_spread_pct is not None else None
                },
                "liquidity_assessment": {
                    "classification": liquidity_classification,
                    "tradability": tradability,
                    "market_impact": impact_assessment,
                    "suggested_max_position": self._suggest_max_position(dollar_volume, avg_hl_spread)
                },
                "notes": "Bid-ask spread is estimated using Roll's model and may not be accurate for all market conditions."
            }
            
            # Cache the result
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_duration
            
            return result
            
        except Exception as e:
            logger.error(f"Error assessing liquidity for {symbol}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to assess liquidity: {str(e)}"
            }
   
    def _suggest_max_position(self, dollar_volume, hl_spread):
        """Suggest maximum position size based on liquidity metrics"""
        # A common rule of thumb is to limit position to ~1-3% of daily volume
        # to minimize market impact
        
        # Adjust for spread - wider spreads suggest smaller positions
        spread_factor = 1.0 / (1.0 + hl_spread)
        
        # Base calculation (conservative)
        max_position = dollar_volume * 0.01 * spread_factor
        
        # Format as string with appropriate units
        if max_position >= 1000000:
            return f"${round(max_position/1000000, 1)}M"
        elif max_position >= 1000:
            return f"${round(max_position/1000, 1)}K"
        else:
            return f"${round(max_position, 2)}"


    # Function to test the market data tools
    # Function to test the market data tools
def test_market_data_tools():
    """Test the market data analysis tools"""
    
    # Get API keys from environment variables
    load_dotenv()
    
    api_keys = {
        'alpha_vantage': os.getenv('ALPHA_VANTAGE_KEY'),
        'polygon': os.getenv('POLYGON_KEY')
    }
    
    # Initialize tools
    market_tools = MarketDataTools(api_keys=api_keys)
    
    # Print API key status
    logger.info(f"Alpha Vantage API Key: {'Provided' if api_keys['alpha_vantage'] else 'Not provided'}")
    logger.info(f"Polygon API Key: {'Provided' if api_keys['polygon'] else 'Not provided'}")
    
    # Test symbol
    test_symbol = "AAPL"
    logger.info(f"\nTesting with symbol: {test_symbol}")
    
    # Test 1: Historical Prices
    logger.info("\n---- Testing Historical Prices ----")
    try:
        prices = market_tools.get_historical_prices(
            symbol=test_symbol,
            interval="1d",
            period="1mo"
        )
        
        if prices["status"] == "success":
            logger.info(f"Retrieved {prices['data_points']} data points")
            logger.info(f"Latest close: {prices['summary']['latest_close']}")
            logger.info(f"Change: {prices['summary']['change']} ({prices['summary']['change_pct']}%)")
            logger.info(f"Average volume: {prices['summary']['avg_volume']}")
        else:
            logger.error(f"Error: {prices['message']}")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
    
    # Test 2: Technical Indicators
    logger.info("\n---- Testing Technical Indicators ----")
    try:
        indicators = market_tools.calculate_technical_indicators(
            symbol=test_symbol,
            indicators=["rsi", "macd", "bollinger"],
            interval="1d",
            period="3mo"
        )
        
        if indicators["status"] == "success":
            logger.info(f"Analysis date: {indicators['as_of_date']}")
            logger.info(f"Overall signal: {indicators['overall_signal']}")
            
            for signal_type, signal in indicators["signals"].items():
                logger.info(f"{signal_type}: {signal}")
        else:
            logger.error(f"Error: {indicators['message']}")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
    
    # Test 3: Market Structure
    logger.info("\n---- Testing Market Structure Analysis ----")
    try:
        structure = market_tools.analyze_market_structure(
            symbol=test_symbol,
            interval="1d",
            period="6mo"
        )
        
        if structure["status"] == "success":
            logger.info(f"Current trend: {structure['market_structure']['trend']}")
            logger.info(f"Key support: {structure['market_structure']['key_support']}")
            logger.info(f"Key resistance: {structure['market_structure']['key_resistance']}")
            
            if structure["market_structure"]["patterns"]:
                logger.info(f"Detected patterns: {', '.join(structure['market_structure']['patterns'])}")
            
            logger.info("\nTechnical outlook:")
            for point in structure["technical_outlook"]:
                logger.info(f"- {point}")
        else:
            logger.error(f"Error: {structure['message']}")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
    
    # Test 4: Volume Profile
    logger.info("\n---- Testing Volume Profile Analysis ----")
    try:
        volume = market_tools.analyze_volume_profile(
            symbol=test_symbol,
            interval="1d",
            period="3mo"
        )
        
        if volume["status"] == "success":
            logger.info(f"Current volume: {volume['volume_analysis']['current_volume']}")
            logger.info(f"Relative volume: {volume['volume_analysis']['relative_volume']}x")
            logger.info(f"Volume trend: {volume['volume_analysis']['volume_trend']}")
            logger.info(f"OBV trend: {volume['obv_trend']}")
            
            logger.info("\nVolume signals:")
            for signal in volume["volume_analysis"]["volume_signals"]:
                logger.info(f"- {signal}")
        else:
            logger.error(f"Error: {volume['message']}")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
    
    # Test 5: Liquidity Assessment
    logger.info("\n---- Testing Liquidity Assessment ----")
    try:
        liquidity = market_tools.assess_liquidity_conditions(
            symbol=test_symbol,
            interval="1d",
            period="3mo"
        )
        
        if liquidity["status"] == "success":
            logger.info(f"Liquidity classification: {liquidity['liquidity_assessment']['classification']}")
            logger.info(f"Tradability: {liquidity['liquidity_assessment']['tradability']}")
            logger.info(f"Avg daily volume: {liquidity['liquidity_metrics']['avg_daily_volume']}")
            logger.info(f"Dollar volume: ${liquidity['liquidity_metrics']['dollar_volume']:,.2f}")
            logger.info(f"Suggested max position: {liquidity['liquidity_assessment']['suggested_max_position']}")
        else:
            logger.error(f"Error: {liquidity['message']}")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")


if __name__ == "__main__":
    # Check for .env file
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("ALPHA_VANTAGE_KEY=\nPOLYGON_KEY=\n")
        logger.info("Created .env file - please add your API keys")
    
    # Import traceback for detailed error reporting
    import traceback
    
    try:
        # Make sure the function call exactly matches the function name
        test_market_data_tools()
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        logger.error(traceback.format_exc())

