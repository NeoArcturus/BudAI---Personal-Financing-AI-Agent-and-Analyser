import time
from mcp.server.fastmcp import FastMCP
from newsdataapi import NewsDataApiClient
import yfinance as yf
import logging
from dotenv import load_dotenv
import os
import warnings
import sys
warnings.filterwarnings("ignore")

load_dotenv()
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

logger = logging.getLogger("uvicorn.error")
mcp = FastMCP("Macro")

# In-memory TTL cache for news queries: {query: (timestamp, formatted_results_str)}
_news_cache = {}
NEWS_CACHE_TTL = 7200  # 2 hours


@mcp.tool()
def get_live_market_data(assets: list[str]) -> str:
    """
    Fetches real-time price, previous close, and percentage change for a list of ticker symbols.
    Example tickers: GBPUSD=X (FX), GC=F (Gold), BZ=F (Brent Crude), ^FTSE (FTSE 100), AAPL (Apple).
    """
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
                    # Fallback for some symbols where fast_info might be sparse
                    current_price = ticker.info.get(
                        'currentPrice') or ticker.info.get('regularMarketPrice')
                    prev_close = ticker.info.get('previousClose') or ticker.info.get(
                        'regularMarketPreviousClose')

                if current_price and prev_close:
                    change_pct = (
                        (current_price - prev_close) / prev_close) * 100
                    results.append(
                        f"Asset: {symbol} | Current Price: {current_price:.4f} | Change: {change_pct:+.2f}%")
                else:
                    results.append(
                        f"Asset: {symbol} | Data currently unavailable via yfinance.")
            except Exception as e:
                results.append(
                    f"Asset: {symbol} | Error fetching data: {str(e)}")

        return "\n".join(results)
    except Exception as e:
        logger.error(f"Macro MCP Error: {e}")
        return f"Error fetching market data: {str(e)}"


@mcp.tool()
def get_financial_news(query: str) -> str:
    """
    Fetches top 3-5 financial and geopolitical news headlines related to the query.
    """
    global _news_cache
    current_time = time.time()

    # Check cache
    if query in _news_cache:
        timestamp, cached_results = _news_cache[query]
        if current_time - timestamp < NEWS_CACHE_TTL:
            logger.info(f"Returning cached news for query: {query}")
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
                snippet = r.get('description') or r.get(
                    'content') or 'No Snippet available.'
                url = r.get('link', '')
                formatted_news.append(
                    f"- {title}\n  Summary: {snippet}\n  Link: {url}")

            final_results = f"Top financial headlines for '{query}':\n" + "\n\n".join(
                formatted_news)

            # Update cache
            _news_cache[query] = (current_time, final_results)

            return final_results
    except Exception as e:
        logger.error(f"News MCP Error: {e}")
        return "Financial news service is currently unavailable."


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sse", action="store_true",
                        help="Run with SSE transport")
    args = parser.parse_args()

    if args.sse:
        logger.info(f"Starting Macro MCP Server with SSE transport")
        mcp.run(transport="sse")
    else:
        mcp.run()
