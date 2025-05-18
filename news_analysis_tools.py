import os
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources (run once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsAnalysisTools:
    """Implementation of tools for news sentiment analysis agent"""
    
    def __init__(self, api_keys=None):
        """Initialize with necessary API keys"""
        self.api_keys = api_keys or {}
        self.news_api_key = self.api_keys.get('news_api', '')
        self.alpha_vantage_key = self.api_keys.get('alpha_vantage', '')
        self.finnhub_key = self.api_keys.get('finnhub', '')
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Cache for expensive API calls
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 30 * 60  # 30 minutes in seconds - shorter for news data
    
    def search_financial_news(self, query, days_back=7, max_results=10):
        """
        Search for recent financial news about a company, sector, or market topic
        
        Args:
            query (str): The search query (company name, ticker, or topic)
            days_back (int): Number of days to look back
            max_results (int): Maximum number of results to return
            
        Returns:
            dict: News articles and sentiment analysis
        """
        try:
            # Check cache first
            cache_key = f"news_search_{query}_{days_back}_{max_results}"
            if cache_key in self.cache and datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
                return self.cache[cache_key]
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')
            
            articles = []
            
            # Try News API if available
            if self.news_api_key:
                try:
                    url = "https://newsapi.org/v2/everything"
                    params = {
                        "q": query,
                        "from": from_date,
                        "to": to_date,
                        "language": "en",
                        "sortBy": "relevancy",
                        "pageSize": max_results,
                        "apiKey": self.news_api_key
                    }
                    
                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status") == "ok":
                            articles = data.get("articles", [])
                            logger.info(f"Retrieved {len(articles)} articles from News API")
                    else:
                        logger.warning(f"News API error: {response.status_code} - {response.text}")
                
                except Exception as e:
                    logger.error(f"Error with News API: {str(e)}")
            
            # Try Alpha Vantage News Sentiment if available and News API failed
            if self.alpha_vantage_key and not articles:
                try:
                    url = "https://www.alphavantage.co/query"
                    params = {
                        "function": "NEWS_SENTIMENT",
                        "tickers": query if len(query) <= 5 else None,  # Assume it's a ticker if ≤ 5 chars
                        "topics": None if len(query) <= 5 else query,   # Otherwise use as topic
                        "time_from": from_date,
                        "limit": max_results,
                        "apikey": self.alpha_vantage_key
                    }
                    
                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        alpha_articles = data.get("feed", [])
                        logger.info(f"Retrieved {len(alpha_articles)} articles from Alpha Vantage")
                        
                        # Convert Alpha Vantage format to our standard format
                        for article in alpha_articles:
                            articles.append({
                                "title": article.get("title"),
                                "description": article.get("summary"),
                                "url": article.get("url"),
                                "publishedAt": article.get("time_published"),
                                "source": {"name": article.get("source")},
                                "sentiment_score": article.get("overall_sentiment_score"),
                                "relevance_score": article.get("relevance_score", 1.0)
                            })
                
                except Exception as e:
                    logger.error(f"Error with Alpha Vantage News API: {str(e)}")
            
            # If no articles found with real APIs, use simulated data
            if not articles:
                logger.info(f"No articles found with available APIs, using simulated data for {query}")
                articles = self._generate_simulated_news(query, days_back, max_results)
            
            # Process articles and analyze sentiment
            processed_articles = []
            sentiment_sum = 0
            sentiment_count = 0
            
            for article in articles:
                # Extract key data
                title = article.get("title", "")
                source = article.get("source", {}).get("name", "Unknown Source")
                published_at = article.get("publishedAt", "")
                url = article.get("url", "")
                description = article.get("description", "")
                
                # Calculate sentiment if not already provided
                if "sentiment_score" in article:
                    sentiment_score = article.get("sentiment_score")
                else:
                    text = title + " " + description
                    sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
                    sentiment_score = sentiment_scores['compound']
                
                # Format date
                formatted_date = published_at
                if published_at:
                    try:
                        dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        formatted_date = dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        pass
                
                # Add to processed articles
                processed_articles.append({
                    "title": title,
                    "source": source,
                    "date": formatted_date,
                    "url": url,
                    "summary": description[:150] + "..." if len(description) > 150 else description,
                    "sentiment_score": round(sentiment_score, 2),
                    "sentiment": self._sentiment_category(sentiment_score)
                })
                
                # Update overall sentiment
                sentiment_sum += sentiment_score
                sentiment_count += 1
            
            # Calculate overall sentiment
            overall_sentiment_score = sentiment_sum / sentiment_count if sentiment_count > 0 else 0
            
            # Prepare result
            result = {
                "status": "success",
                "query": query,
                "articles_found": len(processed_articles),
                "date_range": f"{from_date} to {to_date}",
                "overall_sentiment": {
                    "score": round(overall_sentiment_score, 2),
                    "category": self._sentiment_category(overall_sentiment_score),
                    "interpretation": self._interpret_sentiment(overall_sentiment_score, query)
                },
                "articles": processed_articles,
                "data_source": "News API" if self.news_api_key else 
                               "Alpha Vantage" if self.alpha_vantage_key else 
                               "Simulated Data"
            }
            
            # Cache the result
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_duration
            
            return result
            
        except Exception as e:
            logger.error(f"Error searching for news about {query}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to search for news: {str(e)}"
            }
    
    def _sentiment_category(self, score):
        """Convert sentiment score to category"""
        if score >= 0.5:
            return "Very Positive"
        elif score >= 0.05:
            return "Positive"
        elif score > -0.05:
            return "Neutral"
        elif score > -0.5:
            return "Negative"
        else:
            return "Very Negative"
    
    def _interpret_sentiment(self, score, query):
        """Interpret sentiment score in context"""
        is_ticker = len(query) <= 5 and query.isupper()
        entity_type = "stock" if is_ticker else "topic"
        
        if score >= 0.5:
            return f"Extremely bullish news sentiment for this {entity_type}. Consider this a strongly positive signal."
        elif score >= 0.2:
            return f"Clearly positive news sentiment for this {entity_type}. Media coverage is favorable."
        elif score >= 0.05:
            return f"Mildly positive news sentiment for this {entity_type}. Some favorable coverage."
        elif score > -0.05:
            return f"Neutral news sentiment for this {entity_type}. Mixed or balanced coverage."
        elif score > -0.2:
            return f"Mildly negative news sentiment for this {entity_type}. Some concerning coverage."
        elif score > -0.5:
            return f"Clearly negative news sentiment for this {entity_type}. Media coverage is unfavorable."
        else:
            return f"Extremely bearish news sentiment for this {entity_type}. Consider this a strongly negative signal."
    
    def _generate_simulated_news(self, query, days_back, max_results):
        """Generate simulated news when API is not available"""
        # Create some realistic-looking simulated news
        articles = []
        end_date = datetime.now()
        
        # Different templates based on whether it's likely a stock ticker or general topic
        is_ticker = len(query) <= 5 and query.isupper()
        
        # Possible sentiment biases for simulation
        sentiment_biases = ["positive", "neutral", "negative"]
        selected_bias = sentiment_biases[hash(query) % len(sentiment_biases)]
        
        # Generate simulated articles
        for i in range(max_results):
            days_ago = np.random.randint(0, days_back)
            date = end_date - timedelta(days=days_ago, hours=np.random.randint(0, 24))
            
            if is_ticker:
                # Templates for stock news
                templates = {
                    "positive": [
                        f"{query} Reports Strong Quarterly Earnings, Exceeding Expectations",
                        f"Analysts Upgrade {query} Stock Rating to 'Buy'",
                        f"{query} Announces New Product Line, Stock Surges",
                        f"Institutional Investors Increase Holdings in {query}",
                        f"{query} Signs Major Partnership Deal"
                    ],
                    "neutral": [
                        f"{query} Meets Quarterly Expectations, Maintains Outlook",
                        f"Analysts Maintain Neutral Rating on {query}",
                        f"{query} Announces Management Changes",
                        f"Market Watches {query} Ahead of Industry Conference",
                        f"Understanding {query}'s Position in Current Market"
                    ],
                    "negative": [
                        f"{query} Misses Earnings Expectations, Stock Under Pressure",
                        f"Analysts Downgrade {query} Citing Competitive Pressures",
                        f"{query} Faces Regulatory Scrutiny in Key Markets",
                        f"Supply Chain Issues Impact {query}'s Outlook",
                        f"Investors Concerned About {query}'s Growth Strategy"
                    ]
                }
            else:
                # Templates for general financial topics
                templates = {
                    "positive": [
                        f"Positive Outlook for {query} Sector as Demand Grows",
                        f"New Policies Expected to Boost {query} Growth",
                        f"Investment Opportunities in {query} Highlighted by Experts",
                        f"{query} Shows Strong Growth Potential According to New Study",
                        f"Tech Innovation Drives Optimism in {query} Markets"
                    ],
                    "neutral": [
                        f"{query} Markets Remain Stable Amid Global Uncertainties",
                        f"Understanding the Current State of {query}",
                        f"Experts Discuss Future Trends in {query}",
                        f"{query} Sector Faces Both Opportunities and Challenges",
                        f"Analyzing the Impact of Recent Developments on {query}"
                    ],
                    "negative": [
                        f"Concerns Grow Over {query} Market Stability",
                        f"Challenges Ahead for {query} According to Industry Report",
                        f"Declining Investment Trends in {query} Sector",
                        f"Regulatory Changes May Negatively Impact {query}",
                        f"Economic Uncertainty Clouds Outlook for {query}"
                    ]
                }
            
            # Select template based on sentiment bias
            title_templates = templates[selected_bias]
            title = title_templates[i % len(title_templates)]
            
            # Generate description
            descriptions = {
                "positive": [
                    f"Recent developments indicate strong positive momentum for {query}, with analysts expressing optimism about future prospects.",
                    f"Market sentiment remains favorable for {query}, supported by strong fundamentals and positive industry trends.",
                    f"Investors are showing increased interest in {query} following recent developments that suggest improved growth potential."
                ],
                "neutral": [
                    f"Analysis of {query} shows mixed signals, with both positive and negative factors affecting its outlook.",
                    f"Experts maintain a cautious stance on {query}, citing balanced risk-reward ratio in current market conditions.",
                    f"Recent developments in {query} have not significantly altered market sentiment, maintaining a neutral outlook."
                ],
                "negative": [
                    f"Concerns are growing about {query}'s ability to maintain growth amid challenging market conditions.",
                    f"Recent developments raise questions about {query}'s strategy, with some analysts expressing skepticism.",
                    f"Market sentiment for {query} has deteriorated following reports that highlight potential challenges ahead."
                ]
            }
            
            description_templates = descriptions[selected_bias]
            description = description_templates[i % len(description_templates)]
            
            # Create simulated article
            articles.append({
                "title": title,
                "description": description,
                "url": f"https://example.com/simulated-news/{query.lower().replace(' ', '-')}-{i}",
                "publishedAt": date.isoformat(),
                "source": {"name": f"Simulated Financial News {i+1}"}
            })
        
        return articles
    
    def analyze_company_news_impact(self, ticker, days_back=30):
        """
        Analyze the impact of news on a specific company's stock
        
        Args:
            ticker (str): The company's ticker symbol
            days_back (int): Number of days to analyze
            
        Returns:
            dict: Analysis of news impact on stock price
        """
        try:
            # Check cache
            cache_key = f"news_impact_{ticker}_{days_back}"
            if cache_key in self.cache and datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
                return self.cache[cache_key]
            
            # Get recent news
            news_data = self.search_financial_news(ticker, days_back=days_back, max_results=20)
            
            # Get historical stock data
            historical_prices = None
            price_impact = []
            
            # Try to get real stock data if Alpha Vantage key is available
            if self.alpha_vantage_key:
                try:
                    url = f"https://www.alphavantage.co/query"
                    params = {
                        "function": "TIME_SERIES_DAILY",
                        "symbol": ticker,
                        "outputsize": "compact",
                        "apikey": self.alpha_vantage_key
                    }
                    
                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        time_series = data.get("Time Series (Daily)", {})
                        
                        if time_series:
                            dates = sorted(time_series.keys())
                            prices = []
                            
                            for date in dates:
                                close_price = float(time_series[date]["4. close"])
                                prices.append((date, close_price))
                            
                            historical_prices = prices
                
                except Exception as e:
                    logger.error(f"Error fetching Alpha Vantage stock data: {str(e)}")
            
            # If we couldn't get real stock data, generate simulated data
            if not historical_prices:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                
                # Generate dates
                dates = []
                current_date = start_date
                while current_date <= end_date:
                    if current_date.weekday() < 5:  # Weekdays only
                        dates.append(current_date)
                    current_date += timedelta(days=1)
                
                # Generate simulated prices with some randomness
                base_price = 100  # Starting price
                volatility = 0.02  # Daily volatility
                
                prices = []
                for i, date in enumerate(dates):
                    if i == 0:
                        price = base_price
                    else:
                        change = np.random.normal(0, volatility)
                        price = prices[-1][1] * (1 + change)
                    
                    prices.append((date.strftime('%Y-%m-%d'), price))
                
                historical_prices = prices
            
            # Find news impact on stock prices
            for article in news_data.get("articles", []):
                pub_date = article.get("date")
                sentiment_score = article.get("sentiment_score", 0)
                
                # Find the price movement after the news
                try:
                    pub_datetime = datetime.strptime(pub_date, '%Y-%m-%d %H:%M')
                    pub_date_str = pub_datetime.strftime('%Y-%m-%d')
                    
                    # Find index of this date in prices
                    date_index = None
                    for i, (date, _) in enumerate(historical_prices):
                        if date >= pub_date_str:
                            date_index = i
                            break
                    
                    if date_index is not None and date_index < len(historical_prices) - 1:
                        # Get prices before and after
                        _, price_before = historical_prices[date_index]
                        _, price_after = historical_prices[date_index + 1]
                        
                        # Calculate price change
                        price_change = (price_after - price_before) / price_before * 100
                        
                        # Determine if sentiment aligned with price movement
                        sentiment_aligned = (sentiment_score > 0 and price_change > 0) or \
                                          (sentiment_score < 0 and price_change < 0)
                        
                        price_impact.append({
                            "title": article.get("title"),
                            "date": pub_date,
                            "sentiment_score": sentiment_score,
                            "price_change_pct": round(price_change, 2),
                            "sentiment_aligned": sentiment_aligned
                        })
                        
                except Exception as e:
                    logger.warning(f"Could not analyze price impact for article: {str(e)}")
            
            # Calculate overall statistics
            total_aligned = sum(1 for item in price_impact if item["sentiment_aligned"])
            alignment_rate = total_aligned / len(price_impact) if price_impact else 0
            
            avg_price_change = np.mean([item["price_change_pct"] for item in price_impact]) if price_impact else 0
            avg_sentiment = np.mean([item["sentiment_score"] for item in price_impact]) if price_impact else 0
            
            # Prepare result
            result = {
                "status": "success",
                "ticker": ticker,
                "period_analyzed": f"Last {days_back} days",
                "news_count": len(news_data.get("articles", [])),
                "impact_statistics": {
                    "news_price_correlation": round(alignment_rate, 2),
                    "avg_price_change_after_news": round(avg_price_change, 2),
                    "avg_news_sentiment": round(avg_sentiment, 2),
                    "interpretation": self._interpret_news_impact(alignment_rate, avg_sentiment)
                },
                "notable_impacts": sorted(price_impact, key=lambda x: abs(x["price_change_pct"]), reverse=True)[:5],
                "data_source": "Real stock data (Alpha Vantage)" if self.alpha_vantage_key else "Simulated stock data"
            }
            
            # Cache the result
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_duration
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing news impact for {ticker}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to analyze news impact: {str(e)}"
            }
    
    def _interpret_news_impact(self, alignment_rate, avg_sentiment):
        """Interpret news impact statistics"""
        if alignment_rate >= 0.7:
            importance = "highly influential"
        elif alignment_rate >= 0.5:
            importance = "moderately influential"
        else:
            importance = "minimally influential"
        
        if avg_sentiment > 0.2:
            sentiment = "predominantly positive"
        elif avg_sentiment < -0.2:
            sentiment = "predominantly negative"
        else:
            sentiment = "fairly balanced"
        
        return f"News appears to be {importance} on this stock's price movements, with {sentiment} sentiment overall. "\
               f"News sentiment aligns with subsequent price movements {round(alignment_rate * 100)}% of the time."
    
    def monitor_economic_news(self, topics=None, days_back=7):
        """
        Monitor news about economic indicators and releases
        
        Args:
            topics (list): List of economic topics to monitor (e.g., ["inflation", "interest rates", "gdp"])
            days_back (int): Number of days to look back
            
        Returns:
            dict: Analysis of economic news and potential market implications
        """
        try:
            # Default topics if none provided
            if not topics:
                topics = ["inflation", "interest rates", "fed", "gdp", "unemployment", "recession"]
            
            # Check cache
            topics_key = "_".join(sorted(topics))
            cache_key = f"economic_news_{topics_key}_{days_back}"
            if cache_key in self.cache and datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
                return self.cache[cache_key]
            
            # Collect news for each topic
            topic_news = {}
            for topic in topics:
                news_data = self.search_financial_news(topic, days_back=days_back, max_results=5)
                topic_news[topic] = {
                    "articles": news_data.get("articles", []),
                    "sentiment": news_data.get("overall_sentiment", {})
                }
            
            # Identify most significant economic news
            all_articles = []
            for topic, data in topic_news.items():
                for article in data["articles"]:
                    all_articles.append({
                        "topic": topic,
                        "title": article.get("title"),
                        "source": article.get("source"),
                        "date": article.get("date"),
                        "sentiment_score": article.get("sentiment_score", 0),
                        "url": article.get("url")
                    })
            
            # Sort by absolute sentiment score (most impactful first)
            significant_news = sorted(all_articles, key=lambda x: abs(x.get("sentiment_score", 0)), reverse=True)[:10]
            
            # Analyze overall economic sentiment
            topic_sentiments = {}
            for topic, data in topic_news.items():
                sentiment = data["sentiment"].get("score", 0) if data["sentiment"] else 0
                topic_sentiments[topic] = sentiment
            
            # Overall economic sentiment
            avg_sentiment = np.mean(list(topic_sentiments.values())) if topic_sentiments else 0
            
            # Determine market implications
            market_implications = self._derive_market_implications(topic_sentiments)
            
            # Prepare result
            result = {
                "status": "success",
                "topics_analyzed": topics,
                "significant_economic_news": significant_news,
                "topic_sentiment": {topic: round(score, 2) for topic, score in topic_sentiments.items()},
                "overall_economic_sentiment": {
                    "score": round(avg_sentiment, 2),
                    "category": self._sentiment_category(avg_sentiment),
                    "interpretation": self._interpret_economic_sentiment(avg_sentiment)
                },
                "potential_market_implications": market_implications,
                "data_source": "News API" if self.news_api_key else 
                               "Alpha Vantage" if self.alpha_vantage_key else 
                               "Simulated Data"
            }
            
            # Cache the result
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_duration
            
            return result
            
        except Exception as e:
            logger.error(f"Error monitoring economic news: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to monitor economic news: {str(e)}"
            }
    
    def _interpret_economic_sentiment(self, score):
        """Interpret overall economic sentiment"""
        if score >= 0.3:
            return "Economic news is highly positive, suggesting strong growth prospects and stable conditions."
        elif score >= 0.1:
            return "Economic news is moderately positive, indicating favorable economic conditions with some potential challenges."
        elif score > -0.1:
            return "Economic news is relatively neutral, suggesting stable but potentially mixed economic conditions."
        elif score > -0.3:
            return "Economic news is moderately negative, indicating some economic challenges or concerns."
        else:
            return "Economic news is highly negative, suggesting significant economic headwinds or instability."
    
    def _derive_market_implications(self, topic_sentiments):
        """Derive potential market implications from economic news sentiment"""
        implications = []
        
        # Interest rate implications
        if "interest rates" in topic_sentiments or "fed" in topic_sentiments:
            rate_sentiment = topic_sentiments.get("interest rates", 0) or topic_sentiments.get("fed", 0)
            
            if rate_sentiment > 0.2:
                implications.append("Positive news about interest rates may indicate stable or accommodative monetary policy, potentially supporting equity markets.")
            elif rate_sentiment < -0.2:
                implications.append("Negative news about interest rates could suggest tightening monetary conditions, potentially pressuring equity valuations and growth stocks.")
        
        # Inflation implications
        if "inflation" in topic_sentiments:
            inflation_sentiment = topic_sentiments.get("inflation", 0)
            
            if inflation_sentiment > 0.2:
                implications.append("Positive inflation news may indicate easing price pressures, potentially beneficial for bonds and interest rate sensitive sectors.")
            elif inflation_sentiment < -0.2:
                implications.append("Negative inflation news could suggest persistent price pressures, potentially favoring inflation hedges like commodities and certain value stocks.")
        
        # Growth implications
        if "gdp" in topic_sentiments:
            gdp_sentiment = topic_sentiments.get("gdp", 0)
            
            if gdp_sentiment > 0.2:
                implications.append("Positive GDP news suggests economic strength, potentially favoring cyclical sectors and risk assets.")
            elif gdp_sentiment < -0.2:
                implications.append("Negative GDP news indicates growth concerns, potentially favoring defensive sectors and quality factors.")
        
        # Employment implications
        if "unemployment" in topic_sentiments:
            unemployment_sentiment = topic_sentiments.get("unemployment", 0)
            
            if unemployment_sentiment > 0.2:
                implications.append("Positive employment news indicates labor market strength, potentially supporting consumer sectors and cyclicals.")
            elif unemployment_sentiment < -0.2:
                implications.append("Negative employment news suggests labor market weakness, potentially pressuring consumer discretionary sectors.")
        
        # Recession implications
        if "recession" in topic_sentiments:
            recession_sentiment = topic_sentiments.get("recession", 0)
            
            if recession_sentiment > 0.2:
                implications.append("Positive news regarding recession (e.g., recession fears easing) may improve risk appetite, potentially benefiting cyclical assets.")
            elif recession_sentiment < -0.2:
                implications.append("Negative news about recession risks could drive defensive positioning, potentially benefiting utilities, consumer staples, and quality bonds.")
        
        # If no specific implications, provide general ones
        if not implications:
            # Calculate average sentiment
            avg_sentiment = sum(topic_sentiments.values()) / len(topic_sentiments) if topic_sentiments else 0
            
            if avg_sentiment > 0.1:
                implications.append("Generally positive economic news may support risk assets and cyclical sectors.")
            elif avg_sentiment < -0.1:
                implications.append("Generally negative economic news may favor defensive positioning and safe haven assets.")
            else:
                implications.append("Mixed economic signals suggest a balanced approach to markets with careful sector selection.")
        
        return implications