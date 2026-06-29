import logging
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool
from services.mcp_tools.market_tools_schema import MarketComparisonInput, HistoricalMarketDataInput
from services.mcp_tools.shared_utils import _get_combined_categorized_data
from services.logger_setup import get_core_logger
from datetime import datetime, timedelta

logger = get_core_logger(__name__)

@tool(args_schema=HistoricalMarketDataInput)
def get_historical_market_data(ticker: str, period: str = "6mo", interval: str = "1d") -> str:
    """Fetches historical market data for a given ticker and returns a summary of the trend."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval)
        if hist.empty:
            return f"No historical data found for ticker: {ticker}"
        
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        change_pct = ((end_price - start_price) / start_price) * 100
        
        summary = f"Historical Analysis for {ticker} over {period}:\n"
        summary += f"- Start Price: {start_price:.2f}\n"
        summary += f"- End Price: {end_price:.2f}\n"
        summary += f"- Total Change: {change_pct:+.2f}%\n"
        summary += f"- Volatility (StdDev): {hist['Close'].std():.2f}\n"
        
        return summary
    except Exception as e:
        logger.error(f"Market Tool Error: {e}")
        return f"Error fetching market history: {str(e)}"

@tool(args_schema=MarketComparisonInput)
def compare_spending_to_market(user_uuid: str, category: str, ticker: str, days: int = 180) -> str:
    """Compares a user's spending in a specific category to a market asset's performance.
    Useful for explaining how inflation or commodity prices affect personal finances.
    """
    try:
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
        
        df_spending = _get_combined_categorized_data(["ALL"], "", user_uuid, from_date, to_date)
        if df_spending.empty:
            return "No spending data found for comparison."
        
        df_cat = df_spending[df_spending['Category'].str.lower() == category.lower()]
        if df_cat.empty:
            return f"No spending found in category '{category}' over the last {days} days."
        
        df_cat['date'] = pd.to_datetime(df_cat['date'])
        user_trend = df_cat.set_index('date')['amount'].abs().resample('W').sum()
        
        t = yf.Ticker(ticker)
        hist = t.history(start=from_date, end=to_date)
        if hist.empty:
            return f"Could not fetch market data for {ticker}."
        
        market_trend = hist['Close'].resample('W').mean()
        
        combined = pd.concat([user_trend, market_trend], axis=1).dropna()
        combined.columns = ['User Spending', 'Market Price']
        
        correlation = combined.corr().iloc[0, 1]
        
        market_start = market_trend.iloc[0]
        market_end = market_trend.iloc[-1]
        market_change = ((market_end - market_start) / market_start) * 100
        
        user_start = user_trend.iloc[0]
        user_end = user_trend.iloc[-1]
        user_change = ((user_end - user_start) / user_start) * 100
        
        summary = f"Precision Comparison: '{category}' spending vs {ticker} Performance\n"
        summary += f"- Period: Last {days} days (Weekly Aggregation)\n"
        summary += f"- {ticker} Price Change: {market_change:+.2f}%\n"
        summary += f"- User '{category}' Spend Change: {user_change:+.2f}%\n"
        summary += f"- Statistical Correlation: {correlation:.2f} (1.0 = Perfect Lockstep)\n\n"
        
        if correlation > 0.6:
            summary += f"Observation: Your {category} spending is highly correlated with {ticker} prices. External market factors are likely driving your outflows."
        elif correlation < -0.3:
            summary += f"Observation: Inverse correlation detected. Your spending on {category} decreases as {ticker} prices rise, suggesting elasticity or substitution."
        else:
            summary += f"Observation: Low correlation. Your {category} spending appears independent of {ticker} market fluctuations."
            
        return summary
        
    except Exception as e:
        logger.error(f"Comparison Tool Error: {e}")
        return f"Error performing comparison: {str(e)}"
