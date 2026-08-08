import time
import os
from newsdataapi import NewsDataApiClient
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

from services.logger_setup import get_core_logger
logger = get_core_logger(__name__)


_news_cache = {}
NEWS_CACHE_TTL = 7200

def get_live_market_data(assets: list = None) -> str:
    """
    Fetch live market data and daily performance for a list of asset tickers.
    
    Args:
        assets (list, optional): A list of stock ticker symbols to fetch.
        
    Returns:
        str: A formatted string of current prices and percentage changes.
    """
    if assets is None:
        assets = []
    results = []
    try:
        tickers = yf.Tickers(" ".join(assets))
        for symbol in assets:
            try:
                ticker = tickers.tickers[symbol]
                info = ticker.fast_info
                current_price = info.get('last_price')
                prev_close = info.get('previous_close')

                if current_price is None or prev_close is None:
                    current_price = ticker.info.get('currentPrice') or ticker.info.get('regularMarketPrice')
                    prev_close = ticker.info.get('previousClose') or ticker.info.get('regularMarketPreviousClose')

                if current_price and prev_close:
                    change_pct = ((current_price - prev_close) / prev_close) * 100
                    results.append(f"Asset: {symbol} | Current Price: {current_price:.4f} | Change: {change_pct:+.2f}%")
                else:
                    results.append(f"Asset: {symbol} | Data currently unavailable via yfinance.")
            except Exception as e:
                results.append(f"Asset: {symbol} | Error fetching data: {str(e)}")
        _res = "\n".join(results)
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res
    except Exception as e:
        logger.error(f"Macro Error: {e}")
        _res = f"Error fetching market data: {str(e)}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res

def get_financial_news(query: str = "") -> str:
    """
    Fetch the latest top financial news headlines for a specific query or topic.
    
    Args:
        query (str, optional): The topic or company to search for.
        
    Returns:
        str: A formatted string of recent news headlines with links and summaries.
    """
    global _news_cache
    current_time = time.time()

    if query in _news_cache:
        timestamp, cached_results = _news_cache[query]
        if current_time - timestamp < NEWS_CACHE_TTL:
            _res = cached_results
            logger.info(f"Tool returned: {str(_res)[:1000]}")
            return _res

    try:
        with NewsDataApiClient(apikey=NEWSDATA_API_KEY) as api:
            search_query = f"{query} financial news economy"
            response = api.latest_api(q=search_query, language="en")
            results = response.get('results', [])

            if not results:
                _res = f"No recent financial news found for '{query}'."
                logger.info(f"Tool returned: {str(_res)[:1000]}")
                return _res

            formatted_news = []
            for r in results[:5]:
                title = r.get('title', 'No Title')
                snippet = r.get('description') or r.get('content') or 'No Snippet available.'
                url = r.get('link', '')
                formatted_news.append(f"- {title}\n  Summary: {snippet}\n  Link: {url}")

            final_results = f"Top financial headlines for '{query}':\n" + "\n\n".join(formatted_news)
            _news_cache[query] = (current_time, final_results)
            _res = final_results
            logger.info(f"Tool returned: {str(_res)[:1000]}")
            return _res
    except Exception as e:
        logger.error(f"News Error: {e}")
        _res = "Financial news service is currently unavailable."
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res

def perform_currency_conversion(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert an amount from one currency to another using live exchange rates.
    
    Args:
        amount (float): The amount of money to convert.
        from_currency (str): The origin currency symbol (e.g., USD, GBP).
        to_currency (str): The target currency symbol.
        
    Returns:
        str: The conversion result along with the applied rate.
    """
    ticker_symbol = f"{from_currency}{to_currency}=X"
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.fast_info
        rate = info.get('last_price')
        
        if rate is None:
            rate = ticker.info.get('currentPrice') or ticker.info.get('regularMarketPrice')
            
        if rate:
            converted_amount = amount * rate
            _res = f"CONVERSION_RESULT: {amount} {from_currency} = {converted_amount:.2f} {to_currency} (Rate: {rate:.4f}). [RESULT_VALUE: {converted_amount:.2f}]"
            logger.info(f"Tool returned: {str(_res)[:1000]}")
            return _res
        else:
            _res = f"Could not find exchange rate for {ticker_symbol}. Ensure currency codes are valid."
            logger.info(f"Tool returned: {str(_res)[:1000]}")
            return _res
    except Exception as e:
        logger.error(f"Currency Conversion Error: {e}")
        _res = f"Error performing currency conversion: {str(e)}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res
