
# Fundamental Analysis Agent Tools
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class FundamentalAnalysisTools:
    """Implementation of tools for the fundamental analysis agent"""
    
    def __init__(self, api_keys=None):
        """Initialize with necessary API keys"""
        self.api_keys = api_keys or {}
        # Optional: Initialize connections to financial data providers
        self.sec_api_key = self.api_keys.get('sec_api', '')
        self.alpha_vantage_key = self.api_keys.get('alpha_vantage', '')
    
    def get_financial_statements(self, ticker, statement_type='all', period='annual', years=3):
        """Retrieve company financial statements"""
        try:
            # Use yfinance to get financial data
            company = yf.Ticker(ticker)
            
            statements = {}
            
            if statement_type in ['income', 'all']:
                if period == 'annual':
                    statements['income_statement'] = company.financials
                else:  # quarterly
                    statements['income_statement'] = company.quarterly_financials
            
            if statement_type in ['balance', 'all']:
                if period == 'annual':
                    statements['balance_sheet'] = company.balance_sheet
                else:  # quarterly
                    statements['balance_sheet'] = company.quarterly_balance_sheet
            
            if statement_type in ['cash_flow', 'all']:
                if period == 'annual':
                    statements['cash_flow'] = company.cashflow
                else:  # quarterly
                    statements['cash_flow'] = company.quarterly_cashflow
            
            # Filter to keep only the requested number of years/quarters
            for key in statements:
                if isinstance(statements[key], pd.DataFrame):
                    statements[key] = statements[key].iloc[:, :years]
            
            # Convert DataFrames to dictionaries for easier JSON serialization
            result = {}
            for key, df in statements.items():
                if isinstance(df, pd.DataFrame):
                    result[key] = df.to_dict(orient='index')
                else:
                    result[key] = "Data not available"
            
            return {
                "status": "success",
                "data": result,
                "period": period,
                "years": years
            }
        
        except Exception as e:
            logger.error(f"Error retrieving financial statements for {ticker}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to retrieve financial statements: {str(e)}"
            }
    
    def calculate_financial_ratios(self, ticker, ratio_categories=['all'], benchmark_against=None):
        """Calculate key financial ratios for a company"""
        try:
            # Get company data
            company = yf.Ticker(ticker)
            
            # Get financial statements
            income_statement = company.financials
            balance_sheet = company.balance_sheet
            cash_flow = company.cashflow
            
            # Initialize ratios dictionary
            ratios = {}
            
            # Calculate valuation ratios
            if 'valuation' in ratio_categories or 'all' in ratio_categories:
                # Get market data
                info = company.info
                current_price = info.get('currentPrice', None)
                shares_outstanding = info.get('sharesOutstanding', None)
                market_cap = info.get('marketCap', None)
                
                if current_price and shares_outstanding:
                    # P/E Ratio
                    if 'Net Income' in income_statement.index:
                        net_income = income_statement.loc['Net Income'].iloc[0]
                        ratios['pe_ratio'] = current_price / (net_income / shares_outstanding) if net_income > 0 else None
                    
                    # P/B Ratio
                    if 'Total Stockholder Equity' in balance_sheet.index:
                        book_value = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                        ratios['pb_ratio'] = market_cap / book_value if book_value > 0 else None
                    
                    # EV/EBITDA
                    if 'EBITDA' in income_statement.index:
                        ebitda = income_statement.loc['EBITDA'].iloc[0]
                        total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
                        cash = balance_sheet.loc['Cash'].iloc[0] if 'Cash' in balance_sheet.index else 0
                        enterprise_value = market_cap + total_debt - cash
                        ratios['ev_ebitda'] = enterprise_value / ebitda if ebitda > 0 else None
            
            # Calculate profitability ratios
            if 'profitability' in ratio_categories or 'all' in ratio_categories:
                if 'Total Revenue' in income_statement.index and 'Net Income' in income_statement.index:
                    revenue = income_statement.loc['Total Revenue'].iloc[0]
                    net_income = income_statement.loc['Net Income'].iloc[0]
                    
                    # Net Margin
                    ratios['net_margin'] = net_income / revenue if revenue > 0 else None
                    
                    # ROE
                    if 'Total Stockholder Equity' in balance_sheet.index:
                        equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                        ratios['roe'] = net_income / equity if equity > 0 else None
                    
                    # ROA
                    if 'Total Assets' in balance_sheet.index:
                        assets = balance_sheet.loc['Total Assets'].iloc[0]
                        ratios['roa'] = net_income / assets if assets > 0 else None
            
            # Calculate liquidity ratios
            if 'liquidity' in ratio_categories or 'all' in ratio_categories:
                if 'Current Assets' in balance_sheet.index and 'Current Liabilities' in balance_sheet.index:
                    current_assets = balance_sheet.loc['Current Assets'].iloc[0]
                    current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0]
                    
                    # Current Ratio
                    ratios['current_ratio'] = current_assets / current_liabilities if current_liabilities > 0 else None
                    
                    # Quick Ratio
                    if 'Inventory' in balance_sheet.index:
                        inventory = balance_sheet.loc['Inventory'].iloc[0]
                        quick_assets = current_assets - inventory
                        ratios['quick_ratio'] = quick_assets / current_liabilities if current_liabilities > 0 else None
            
            # Add benchmarking if requested
            if benchmark_against:
                competitor_ratios = {}
                for comp_ticker in benchmark_against:
                    try:
                        # Simplified - in a real implementation, you'd reuse the ratio calculation logic
                        comp = yf.Ticker(comp_ticker)
                        comp_info = comp.info
                        
                        # Just calculate a couple of key ratios for the example
                        comp_ratios = {}
                        if 'trailingPE' in comp_info:
                            comp_ratios['pe_ratio'] = comp_info['trailingPE']
                        if 'priceToBook' in comp_info:
                            comp_ratios['pb_ratio'] = comp_info['priceToBook']
                        
                        competitor_ratios[comp_ticker] = comp_ratios
                    
                    except Exception as e:
                        competitor_ratios[comp_ticker] = f"Error: {str(e)}"
                
                return {
                    "status": "success",
                    "company_ratios": ratios,
                    "benchmark_ratios": competitor_ratios
                }
            
            return {
                "status": "success",
                "ratios": ratios
            }
            
        except Exception as e:
            logger.error(f"Error calculating financial ratios for {ticker}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to calculate financial ratios: {str(e)}"
            }
    
    def analyze_earnings_reports(self, ticker, quarters_back=4, include_call_transcripts=False):
        """Extract and analyze data from recent earnings reports"""
        try:
            # Get earnings data from yfinance
            company = yf.Ticker(ticker)
            earnings = company.earnings
            quarterly_earnings = company.quarterly_earnings
            
            # Get analyst estimates
            if hasattr(company, 'analyst_recommendation_trend'):
                analyst_recommendations = company.analyst_recommendation_trend
            else:
                analyst_recommendations = None
            
            # For call transcripts, we would need a specialized API
            call_transcript_analysis = None
            if include_call_transcripts:
                # This would require a specialized provider like Alpha Vantage Premium, Seeking Alpha, etc.
                # For demonstration, we'll return a placeholder
                call_transcript_analysis = {
                    "available": False,
                    "message": "Earnings call transcript analysis requires integration with a specialized provider"
                }
            
            # Format quarterly data, limiting to requested number of quarters
            quarters_data = {}
            if isinstance(quarterly_earnings, pd.DataFrame):
                quarterly_earnings = quarterly_earnings.iloc[:quarters_back]
                quarters_data = quarterly_earnings.to_dict()
            
            # Calculate earnings surprises
            surprises = {}
            if hasattr(company, 'earnings_dates') and isinstance(company.earnings_dates, pd.DataFrame):
                surprises_df = company.earnings_dates.iloc[:quarters_back]
                if 'Surprise(%)' in surprises_df.columns:
                    surprises = surprises_df['Surprise(%)'].to_dict()
            
            return {
                "status": "success",
                "annual_earnings": earnings.to_dict() if isinstance(earnings, pd.DataFrame) else {},
                "quarterly_earnings": quarters_data,
                "earnings_surprises": surprises,
                "analyst_recommendations": analyst_recommendations.to_dict() if isinstance(analyst_recommendations, pd.DataFrame) else {},
                "call_transcript_analysis": call_transcript_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing earnings reports for {ticker}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to analyze earnings reports: {str(e)}"
            }
    
    def get_industry_analysis(self, ticker, metrics=['all'], industry_code=None, include_competitors=True):
        """Retrieve industry data and competitive positioning"""
        try:
            # Get company info to determine industry
            company = yf.Ticker(ticker)
            info = company.info
            
            industry = info.get('industry', None)
            sector = info.get('sector', None)
            
            # For a real implementation, you would use a database or API with industry classifications
            # like GICS, SIC, or NAICS codes to get more detailed industry data
            
            # Find competitors in the same industry
            competitors = []
            if include_competitors:
                # This is a simplified approach - in practice, you would use a more comprehensive database
                if industry:
                    # You might use an API or database to find companies in the same industry
                    # For demonstration, we'll just return a placeholder
                    competitors = ["Would connect to industry database to find competitors"]
            
            # Industry metrics
            industry_metrics = {}
            
            # Size metrics
            if 'size' in metrics or 'all' in metrics:
                industry_metrics['size'] = {
                    "total_market_cap": "Would retrieve from industry database",
                    "largest_companies": "Would retrieve top 5 companies by market cap"
                }
            
            # Growth metrics
            if 'growth' in metrics or 'all' in metrics:
                industry_metrics['growth'] = {
                    "revenue_growth_rate": "Would calculate from industry database",
                    "market_expansion_rate": "Would retrieve from economic data"
                }
            
            # Concentration metrics
            if 'concentration' in metrics or 'all' in metrics:
                industry_metrics['concentration'] = {
                    "herfindahl_index": "Would calculate from industry market share data",
                    "top_5_concentration_ratio": "Would calculate from industry data"
                }
            
            return {
                "status": "success",
                "company": {
                    "ticker": ticker,
                    "industry": industry,
                    "sector": sector
                },
                "industry_metrics": industry_metrics,
                "competitors": competitors
            }
            
        except Exception as e:
            logger.error(f"Error retrieving industry analysis for {ticker}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to retrieve industry analysis: {str(e)}"
            }
    
    def run_valuation_model(self, ticker, models=['dcf'], forecast_years=5, growth_assumptions=None):
        """Run financial valuation models on a company"""
        try:
            # Get company data
            company = yf.Ticker(ticker)
            info = company.info
            financials = company.financials
            
            # Get current financial data
            current_price = info.get('currentPrice', None)
            shares_outstanding = info.get('sharesOutstanding', None)
            market_cap = current_price * shares_outstanding if current_price and shares_outstanding else None
            
            # Default growth assumptions if not provided
            if not growth_assumptions:
                growth_assumptions = {
                    "revenue_growth": 0.05,  # 5% annual growth
                    "margin_growth": 0.01,   # 1% margin improvement
                    "terminal_growth": 0.02  # 2% terminal growth
                }
            
            # Valuation results
            valuation_results = {}
            
            # Discounted Cash Flow (DCF) Model
            if 'dcf' in models or 'all' in models:
                try:
                    # In a real implementation, this would be much more sophisticated
                    # This is a very simplified DCF model for demonstration
                    
                    # Get free cash flow from most recent year
                    if 'Free Cash Flow' in financials.index:
                        latest_fcf = financials.loc['Free Cash Flow'].iloc[0]
                    else:
                        # Estimate FCF from operating cash flow and capital expenditures
                        cash_flow = company.cashflow
                        op_cash_flow = cash_flow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cash_flow.index else 0
                        capex = cash_flow.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cash_flow.index else 0
                        latest_fcf = op_cash_flow + capex  # Capex is typically negative
                    
                    # Project future cash flows
                    discount_rate = 0.10  # 10% discount rate
                    fcf_projections = []
                    
                    for year in range(1, forecast_years + 1):
                        growth_rate = growth_assumptions['revenue_growth']
                        projected_fcf = latest_fcf * (1 + growth_rate) ** year
                        fcf_projections.append(projected_fcf)
                    
                    # Calculate terminal value
                    terminal_growth = growth_assumptions['terminal_growth']
                    terminal_value = fcf_projections[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
                    
                    # Discount future cash flows
                    dcf_value = 0
                    for i, fcf in enumerate(fcf_projections):
                        dcf_value += fcf / (1 + discount_rate) ** (i + 1)
                    
                    # Add discounted terminal value
                    dcf_value += terminal_value / (1 + discount_rate) ** forecast_years
                    
                    # Calculate per-share value
                    dcf_per_share = dcf_value / shares_outstanding if shares_outstanding else None
                    
                    valuation_results['dcf'] = {
                        "enterprise_value": dcf_value,
                        "per_share_value": dcf_per_share,
                        "upside_potential": (dcf_per_share / current_price - 1) * 100 if current_price and dcf_per_share else None,
                        "assumptions": {
                            "discount_rate": discount_rate,
                            "forecast_years": forecast_years,
                            "growth_assumptions": growth_assumptions
                        }
                    }
                except Exception as e:
                    valuation_results['dcf'] = {
                        "status": "error",
                        "message": f"DCF calculation failed: {str(e)}"
                    }
            
            # Comparable Companies Analysis
            if 'comparable_companies' in models or 'all' in models:
                # In a real implementation, you would fetch data for peer companies
                # and calculate average multiples
                valuation_results['comparable_companies'] = {
                    "message": "Comparable companies analysis requires industry peer data",
                    "implementation_note": "Would fetch P/E, EV/EBITDA, P/S ratios for industry peers and apply to this company"
                }
            
            return {
                "status": "success",
                "company": {
                    "ticker": ticker,
                    "current_price": current_price,
                    "market_cap": market_cap
                },
                "valuation_results": valuation_results
            }
            
        except Exception as e:
            logger.error(f"Error running valuation model for {ticker}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to run valuation model: {str(e)}"
            }
    
    def get_esg_metrics(self, ticker, categories=['all'], include_industry_comparison=False):
        """Retrieve ESG metrics for a company"""
        try:
            # In a production environment, you would use a specialized ESG data provider
            # like MSCI, Sustainalytics, or Bloomberg ESG
            
            # For demonstration, we'll simulate ESG data
            company = yf.Ticker(ticker)
            info = company.info
            
            company_name = info.get('shortName', ticker)
            
            # Simulated ESG data - in reality, this would come from an ESG data provider API
            esg_data = {
                "environmental": {
                    "carbon_emissions": {
                        "score": 65,  # 0-100 score
                        "trend": "improving",
                        "details": "Score represents carbon intensity relative to industry peers"
                    },
                    "resource_use": {
                        "score": 70,
                        "trend": "stable",
                        "details": "Measures efficiency of energy, water, and material usage"
                    }
                },
                "social": {
                    "human_capital": {
                        "score": 72,
                        "trend": "improving",
                        "details": "Evaluates labor practices, employee health & safety, diversity"
                    },
                    "community_relations": {
                        "score": 68,
                        "trend": "stable",
                        "details": "Assesses community engagement and impact"
                    }
                },
                "governance": {
                    "board_quality": {
                        "score": 80,
                        "trend": "improving",
                        "details": "Evaluates board independence, diversity, compensation practices"
                    },
                    "business_ethics": {
                        "score": 75,
                        "trend": "stable",
                        "details": "Assesses anti-corruption measures, ethical business practices"
                    }
                },
                "controversies": {
                    "severity": "low",
                    "count": 2,
                    "details": "Minor environmental compliance issues reported in past year"
                }
            }
            
            # Filter to requested categories
            if 'all' not in categories:
                filtered_data = {}
                for category in categories:
                    if category in esg_data:
                        filtered_data[category] = esg_data[category]
                esg_data = filtered_data
            
            # Add industry comparison if requested
            if include_industry_comparison:
                # In reality, this would come from industry benchmark data
                industry_comparison = {
                    "industry_average": {
                        "environmental": 60,
                        "social": 65,
                        "governance": 70,
                        "overall": 65
                    },
                    "percentile_rank": {
                        "environmental": 70,  # Better than 70% of peers
                        "social": 65,
                        "governance": 80,
                        "overall": 72
                    }
                }
            else:
                industry_comparison = None
            
            return {
                "status": "success",
                "company": company_name,
                "ticker": ticker,
                "esg_data": esg_data,
                "industry_comparison": industry_comparison,
                "data_source": "Simulated data - would use actual ESG provider in production"
            }
            
        except Exception as e:
            logger.error(f"Error retrieving ESG metrics for {ticker}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to retrieve ESG metrics: {str(e)}"
            }
    
    def analyze_insider_transactions(self, ticker, months_back=6, transaction_types=['all']):
        """Analyze recent insider buying and selling patterns"""
        try:
            # This would normally use an API that provides insider transaction data
            # such as SEC Form 4 filings data
            
            # For demonstration, we'll create sample insider transaction data
            today = datetime.now()
            start_date = today - timedelta(days=months_back * 30)
            
            # In reality, you would fetch this data from an SEC filings API
            # This is simulated data for demonstration
            insider_transactions = [
                {
                    "date": (today - timedelta(days=15)).strftime("%Y-%m-%d"),
                    "insider_name": "John Smith",
                    "title": "CEO",
                    "transaction_type": "purchase",
                    "shares": 5000,
                    "price_per_share": 150.25,
                    "total_value": 751250
                },
                {
                    "date": (today - timedelta(days=30)).strftime("%Y-%m-%d"),
                    "insider_name": "Jane Doe",
                    "title": "CFO",
                    "transaction_type": "sale",
                    "shares": 2000,
                    "price_per_share": 148.50,
                    "total_value": 297000
                },
                {
                    "date": (today - timedelta(days=45)).strftime("%Y-%m-%d"),
                    "insider_name": "Robert Johnson",
                    "title": "Director",
                    "transaction_type": "purchase",
                    "shares": 1000,
                    "price_per_share": 145.75,
                    "total_value": 145750
                }
            ]
            
            # Filter by transaction type if specified
            if 'all' not in transaction_types:
                insider_transactions = [t for t in insider_transactions if t['transaction_type'] in transaction_types]
            
            # Calculate summary statistics
            total_purchases = sum(t['shares'] for t in insider_transactions if t['transaction_type'] == 'purchase')
            total_sales = sum(t['shares'] for t in insider_transactions if t['transaction_type'] == 'sale')
            net_activity = total_purchases - total_sales
            
            # Get transaction by insider role
            transactions_by_role = {}
            for transaction in insider_transactions:
                role = transaction['title']
                if role not in transactions_by_role:
                    transactions_by_role[role] = []
                transactions_by_role[role].append(transaction)
            
            return {
                "status": "success",
                "ticker": ticker,
                "insider_transactions": insider_transactions,
                "summary": {
                    "total_transactions": len(insider_transactions),
                    "total_purchase_shares": total_purchases,
                    "total_sale_shares": total_sales,
                    "net_activity": net_activity,
                    "net_activity_signal": "bullish" if net_activity > 0 else "bearish" if net_activity < 0 else "neutral"
                },
                "transactions_by_role": transactions_by_role,
                "date_range": {
                    "start": start_date.strftime("%Y-%m-%d"),
                    "end": today.strftime("%Y-%m-%d")
                },
                "data_source": "Simulated data - would use SEC filings API in production"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing insider transactions for {ticker}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to analyze insider transactions: {str(e)}"
            }
    
    def get_analyst_estimates(self, ticker, estimate_type=['all'], time_period='all', include_history=False):
        """Retrieve analyst estimates, ratings, and price targets"""
        try:
            company = yf.Ticker(ticker)
            
            # Get current price
            current_price = company.info.get('currentPrice', None)
            
            # Get analyst recommendations
            analyst_recommendations = None
            if hasattr(company, 'recommendations'):
                analyst_recommendations = company.recommendations
                
            # Get earnings estimates
            earnings_estimates = None
            if hasattr(company, 'earnings_forecasts'):
                earnings_estimates = company.earnings_forecasts
                
            # Get revenue estimates - this might not be directly available in yfinance
            # For a production implementation, you would use a more comprehensive data source
            
            # Get analyst price targets - might need to be extrapolated from recommendations
            latest_price_targets = {}
            if isinstance(analyst_recommendations, pd.DataFrame) and 'Price Target' in analyst_recommendations.columns:
                latest_price_targets = analyst_recommendations['Price Target'].sort_index(ascending=False).head(10).to_dict()
            
            # Format analyst ratings
            ratings_summary = {}
            if isinstance(analyst_recommendations, pd.DataFrame) and 'To Grade' in analyst_recommendations.columns:
                # Count ratings by category
                ratings = analyst_recommendations['To Grade'].value_counts().to_dict()
                
                # Calculate consensus rating
                bullish_ratings = sum(ratings.get(grade, 0) for grade in ['Buy', 'Strong Buy', 'Outperform', 'Overweight'])
                neutral_ratings = sum(ratings.get(grade, 0) for grade in ['Hold', 'Neutral', 'Perform', 'Market Perform'])
                bearish_ratings = sum(ratings.get(grade, 0) for grade in ['Sell', 'Underperform', 'Underweight'])
                
                total_ratings = bullish_ratings + neutral_ratings + bearish_ratings
                
                if total_ratings > 0:
                    consensus = "Bullish" if bullish_ratings / total_ratings > 0.6 else \
                               "Bearish" if bearish_ratings / total_ratings > 0.4 else "Neutral"
                else:
                    consensus = "No consensus available"
                
                ratings_summary = {
                    "ratings_breakdown": ratings,
                    "bullish_count": bullish_ratings,
                    "neutral_count": neutral_ratings,
                    "bearish_count": bearish_ratings,
                    "consensus": consensus
                }
            
            # Filter by estimate type and time period
            result = {
                "status": "success",
                "ticker": ticker,
                "current_price": current_price
            }
            
            if 'eps' in estimate_type or 'all' in estimate_type:
                result["earnings_estimates"] = earnings_estimates.to_dict() if isinstance(earnings_estimates, pd.DataFrame) else "Not available"
            
            if 'ratings' in estimate_type or 'all' in estimate_type:
                result["analyst_ratings"] = ratings_summary
            
            if 'price_target' in estimate_type or 'all' in estimate_type:
                # Calculate average, high, low price targets
                if latest_price_targets:
                    pt_values = list(latest_price_targets.values())
                    avg_pt = sum(pt_values) / len(pt_values)
                    high_pt = max(pt_values)
                    low_pt = min(pt_values)
                    
                    # Calculate upside potential
                    upside = (avg_pt / current_price - 1) * 100 if current_price else None
                    
                    result["price_targets"] = {
                        "average": avg_pt,
                        "high": high_pt,
                        "low": low_pt,
                        "upside_potential_pct": upside
                    }
                else:
                    result["price_targets"] = "Not available"
            
            if include_history:
                # Historical estimate accuracy would require comparing past estimates to actual results
                # This would typically require a more specialized data source
                result["historical_accuracy"] = {
                    "message": "Historical accuracy analysis requires specialized data sources",
                    "implementation_note": "Would compare past estimates to actual results to assess analyst accuracy"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving analyst estimates for {ticker}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to retrieve analyst estimates: {str(e)}"
            }
    
    def assess_fundamental_risks(self, ticker, risk_categories=['all'], include_mitigation=False):
        """Identify and assess fundamental business risks"""
        try:
            company = yf.Ticker(ticker)
            info = company.info
            
            company_name = info.get('shortName', ticker)
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            # In a real implementation, this would analyze data from multiple sources:
            # - SEC filings (risk factors section)
            # - Financial ratios (leverage, liquidity)
            # - Industry reports
            # - News and sentiment data
            
            # For demonstration, we'll create simulated risk assessments
            risk_assessment = {
                "financial": {
                    "leverage_risk": {
                        "risk_level": "moderate",
                        "description": "Debt-to-EBITDA ratio slightly above industry average",
                        "impact": "Moderate impact on financial flexibility and interest coverage",
                        "mitigation": "Company has staggered debt maturity schedule reducing refinancing risk"
                    },
                    "liquidity_risk": {
                        "risk_level": "low",
                        "description": "Strong current ratio and cash position",
                        "impact": "Well-positioned to meet short-term obligations",
                        "mitigation": "Maintained revolving credit facility providing additional liquidity buffer"
                    }
                },
                "operational": {
                    "supply_chain_risk": {
                        "risk_level": "high",
                        "description": "Heavy reliance on single-source suppliers for key components",
                        "impact": "Production disruptions possible if supply chain issues occur",
                        "mitigation": "Beginning supplier diversification program, increasing inventory buffers"
                    },
                    "geographic_concentration": {
                        "risk_level": "moderate",
                        "description": "Manufacturing facilities concentrated in one region",
                        "impact": "Regional disruptions could significantly impact production",
                        "mitigation": "Insurance coverage for business interruption, disaster recovery plans in place"
                    }
                },
                "competitive": {
                    "market_share_risk": {
                        "risk_level": "moderate",
                        "description": "Increasing competition from emerging players",
                        "impact": "Potential margin pressure and gradual market share erosion",
                       "mitigation": "Investing in product differentiation and customer loyalty programs"
                   },
                   "technology_disruption": {
                       "risk_level": "high",
                       "description": "Industry facing potential technological disruption",
                       "impact": "Current business model could become obsolete in 3-5 years",
                       "mitigation": "R&D investments in next-gen technology, strategic acquisitions of innovative startups"
                   }
               },
               "legal": {
                   "regulatory_risk": {
                       "risk_level": "moderate",
                       "description": "Industry facing increased regulatory scrutiny",
                       "impact": "Compliance costs likely to increase, potential fines for violations",
                       "mitigation": "Enhanced compliance program, engagement with regulatory bodies"
                   },
                   "litigation_risk": {
                       "risk_level": "low",
                       "description": "Limited history of significant litigation",
                       "impact": "No material impact on financials expected",
                       "mitigation": "Strong legal team, product safety protocols"
                   }
               },
               "market": {
                   "interest_rate_risk": {
                       "risk_level": "moderate",
                       "description": "Floating rate debt exposure in rising rate environment",
                       "impact": "Increased interest expense could pressure margins",
                       "mitigation": "Interest rate hedging program covering 60% of debt"
                   },
                   "foreign_exchange_risk": {
                       "risk_level": "high",
                       "description": "Significant revenue from international markets",
                       "impact": "Earnings volatility due to currency fluctuations",
                       "mitigation": "Natural hedging through local costs, selective use of currency forwards"
                   }
               }
           }
           
            # Filter by risk categories if specified
            if 'all' not in risk_categories:
                filtered_risks = {}
                for category in risk_categories:
                    if category in risk_assessment:
                        filtered_risks[category] = risk_assessment[category]
                risk_assessment = filtered_risks
            
            # Remove mitigation information if not requested
            if not include_mitigation:
                for category in risk_assessment:
                    for risk in risk_assessment[category]:
                        if 'mitigation' in risk_assessment[category][risk]:
                            del risk_assessment[category][risk]['mitigation']
            
            # Calculate overall risk score (simplified approach)
            risk_levels = {
                "low": 1,
                "moderate": 2,
                "high": 3
            }
            
            risk_scores = []
            for category in risk_assessment:
                for risk in risk_assessment[category]:
                    risk_level = risk_assessment[category][risk]['risk_level']
                    risk_scores.append(risk_levels.get(risk_level, 0))
            
            avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0
            overall_risk = "low" if avg_risk_score < 1.5 else "moderate" if avg_risk_score < 2.5 else "high"
            
            # Identify top risks
            top_risks = []
            for category in risk_assessment:
                for risk_name, risk_data in risk_assessment[category].items():
                    if risk_data['risk_level'] == 'high':
                        top_risks.append({
                            "category": category,
                            "risk_name": risk_name,
                            "description": risk_data['description'],
                            "impact": risk_data['impact']
                        })
            
            return {
                "status": "success",
                "company": company_name,
                "ticker": ticker,
                "industry": industry,
                "sector": sector,
                "overall_risk_assessment": overall_risk,
                "top_risks": top_risks,
                "detailed_risk_assessment": risk_assessment,
                "data_source": "Simulated data - would use SEC filings, financial analysis in production"
            }
            
        except Exception as e:
            logger.error(f"Error assessing fundamental risks for {ticker}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to assess fundamental risks: {str(e)}"
            }
    
    def parse_sec_filings(self, ticker, filing_types=['10-K'], items_to_extract=['all'], filing_date_range='last_year'):
        """Extract and analyze key information from SEC filings"""
        try:
            # In a production environment, you would use an SEC API or service like SEC EDGAR
            # For demonstration, we'll simulate this with placeholder data
            
            company = yf.Ticker(ticker)
            info = company.info
            
            company_name = info.get('shortName', ticker)
            
            # Parse the date range
            today = datetime.now()
            if filing_date_range == 'last_year':
                start_date = (today - timedelta(days=365)).strftime("%Y-%m-%d")
                end_date = today.strftime("%Y-%m-%d")
            elif filing_date_range == 'last_quarter':
                start_date = (today - timedelta(days=90)).strftime("%Y-%m-%d")
                end_date = today.strftime("%Y-%m-%d")
            elif ':' in filing_date_range:
                # Custom date range format: 'YYYY-MM-DD:YYYY-MM-DD'
                dates = filing_date_range.split(':')
                start_date = dates[0]
                end_date = dates[1] if len(dates) > 1 else today.strftime("%Y-%m-%d")
            else:
                # Default to last year
                start_date = (today - timedelta(days=365)).strftime("%Y-%m-%d")
                end_date = today.strftime("%Y-%m-%d")
            
            # Simulated SEC filing data
            sec_filings = {
                "10-K": {
                    "filing_date": (today - timedelta(days=60)).strftime("%Y-%m-%d"),
                    "period_end": (today - timedelta(days=90)).strftime("%Y-%m-%d"),
                    "url": f"https://www.sec.gov/edgar/example-10k-url/{ticker}",
                    "items": {
                        "risk_factors": {
                            "summary": "Key risks include technology disruption, competitive pressure, and cybersecurity threats",
                            "significant_changes": "Added new risk factors related to supply chain disruptions",
                            "extract": "The company faces significant risks from rapidly evolving technology..."
                        },
                        "management_discussion": {
                            "summary": "Management highlights revenue growth of 12% YoY driven by new product lines",
                            "key_metrics": "Gross margin improved 150 basis points to 42.5%",
                            "extract": "Our strategic initiatives delivered strong results despite macroeconomic headwinds..."
                        },
                        "legal_proceedings": {
                            "summary": "One material lawsuit related to patent infringement",
                            "potential_impact": "Estimated potential liability of $10-15M if unsuccessful",
                            "extract": "The company is currently defending against claims that certain technologies..."
                        }
                    }
                },
                "10-Q": {
                    "filing_date": (today - timedelta(days=15)).strftime("%Y-%m-%d"),
                    "period_end": (today - timedelta(days=45)).strftime("%Y-%m-%d"),
                    "url": f"https://www.sec.gov/edgar/example-10q-url/{ticker}",
                    "items": {
                        "financial_statements": {
                            "summary": "Quarterly revenue of $1.25B, up 8% YoY",
                            "key_metrics": "EPS of $1.12, beating consensus estimates by $0.08",
                            "extract": "The company reported strong financial performance for the quarter..."
                        },
                        "risk_factors": {
                            "summary": "No material changes to risk factors since last 10-K",
                            "extract": "There have been no material changes to our risk factors..."
                        }
                    }
                }
            }
            
            # Filter by filing types
            if 'all' not in filing_types:
                filtered_filings = {}
                for filing_type in filing_types:
                    if filing_type in sec_filings:
                        filtered_filings[filing_type] = sec_filings[filing_type]
                sec_filings = filtered_filings
            
            # Filter by items to extract
            if 'all' not in items_to_extract:
                for filing_type in sec_filings:
                    if 'items' in sec_filings[filing_type]:
                        filtered_items = {}
                        for item in items_to_extract:
                            if item in sec_filings[filing_type]['items']:
                                filtered_items[item] = sec_filings[filing_type]['items'][item]
                        sec_filings[filing_type]['items'] = filtered_items
            
            # Prepare the response
            extracted_data = {}
            for filing_type, filing_data in sec_filings.items():
                if filing_data.get('filing_date') >= start_date and filing_data.get('filing_date') <= end_date:
                    extracted_data[filing_type] = filing_data
            
            return {
                "status": "success",
                "company": company_name,
                "ticker": ticker,
                "date_range": {
                    "start": start_date,
                    "end": end_date
                },
                "extracted_data": extracted_data,
                "data_source": "Simulated data - would use SEC EDGAR API in production"
            }
            
        except Exception as e:
            logger.error(f"Error parsing SEC filings for {ticker}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to parse SEC filings: {str(e)}"
            }