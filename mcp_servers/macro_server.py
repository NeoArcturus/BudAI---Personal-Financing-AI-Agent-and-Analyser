import time
import asyncio
import os
import warnings
import sys
from iii import register_worker
from newsdataapi import NewsDataApiClient
import yfinance as yf
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")
III_ENGINE_URL = os.getenv("III_ENGINE_URL", "ws://iii-engine:49134")

from services.logger_setup import get_core_logger
logger = get_core_logger(__name__)

_news_cache = {}
NEWS_CACHE_TTL = 7200

def get_live_market_data(assets: list = None) -> str:
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
        return "\n".join(results)
    except Exception as e:
        logger.error(f"Macro Error: {e}")
        return f"Error fetching market data: {str(e)}"

def get_financial_news(query: str = "") -> str:
    global _news_cache
    current_time = time.time()

    if query in _news_cache:
        timestamp, cached_results = _news_cache[query]
        if current_time - timestamp < NEWS_CACHE_TTL:
            return cached_results

    try:
        with NewsDataApiClient(apikey=NEWSDATA_API_KEY) as api:
            search_query = f"{query} financial news economy"
            response = api.latest_api(q=search_query, language="en")
            results = response.get('results', [])

            if not results:
                return f"No recent financial news found for '{query}'."

            formatted_news = []
            for r in results[:5]:
                title = r.get('title', 'No Title')
                snippet = r.get('description') or r.get('content') or 'No Snippet available.'
                url = r.get('link', '')
                formatted_news.append(f"- {title}\n  Summary: {snippet}\n  Link: {url}")

            final_results = f"Top financial headlines for '{query}':\n" + "\n\n".join(formatted_news)
            _news_cache[query] = (current_time, final_results)
            return final_results
    except Exception as e:
        logger.error(f"News Error: {e}")
        return "Financial news service is currently unavailable."

def tool_wrapper(func, args):
    # Filter out internal iii arguments
    clean_args = {k: v for k, v in args.items() if not k.startswith("_")}
    try:
        if asyncio.iscoroutinefunction(func):
            try:
                return asyncio.run(func(**clean_args))
            except RuntimeError:
                # Fallback if loop is already running
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(func(**clean_args))
        else:
            return func(**clean_args)
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {e}", exc_info=True)
        return f"Error: {str(e)}"

async def main():
    logger.info("Initializing Macro Worker with iii")
    worker = register_worker(III_ENGINE_URL)
    
    worker.register_function("macro::get_live_market_data", lambda args: tool_wrapper(get_live_market_data, args))
    worker.register_function("macro::get_financial_news", lambda args: tool_wrapper(get_financial_news, args))
    
    logger.info("Macro Worker registered. Entering heartbeat loop.")
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
