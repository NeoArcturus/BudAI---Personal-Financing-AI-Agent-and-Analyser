import json
from fastapi import HTTPException
from models.database_models import User
import yfinance as yf
from newsdataapi import NewsDataApiClient
from config import NEWSDATA_API_KEY
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

async def get_market_ticker(current_user: User):
    """
    Fetches real-time price and daily change percentage for a predefined set of key market assets.
    
    Args:
        current_user (User): The authenticated user making the request.
        
    Returns:
        dict: A payload containing ticker symbols, current prices, and % changes.
        
    Raises:
        HTTPException (500): If fetching market data fails.
    """
    assets = ["GBPUSD=X", "GC=F", "BZ=F", "^FTSE",
              "^GSPC", "SI=F", "BTC-USD", "^IXIC", "EURGBP=X"]
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
                    results.append({
                        "symbol": symbol,
                        "price": round(current_price, 4),
                        "change": round(change_pct, 2)
                    })
            except Exception as e:
                logger.warning(json.dumps({"message": f"Failed to fetch ticker {symbol}: {e}", "status_code": 400}))
        return {"tickers": results}
    except Exception as e:
        logger.error(json.dumps({"message": f"Market ticker failed: {e}", "status_code": 500}))
        raise HTTPException(
            status_code=500, detail="Failed to fetch market data")

async def get_market_history(range: str, current_user: User):
    """
    Retrieves historical closing prices for predefined market assets over a specified time range.
    
    Args:
        range (str): The requested time period (e.g., '1D', '1W', '1M', '1Y').
        current_user (User): The authenticated user making the request.
        
    Returns:
        dict: A payload containing historical price points per asset.
        
    Raises:
        HTTPException (500): If fetching market history fails.
    """
    range_map = {
        "1D": "1d",
        "1W": "5d",
        "1M": "1mo",
        "3M": "3mo",
        "6M": "6mo",
        "1Y": "1y",
    }
    period = range_map.get(range, "1mo")
    assets = ["GBPUSD=X", "GC=F", "BZ=F", "^FTSE",
              "^GSPC", "SI=F", "BTC-USD", "^IXIC", "EURGBP=X"]
    results = []
    try:
        for symbol in assets:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if not hist.empty:
                    data_points = []
                    for date, row in hist.iterrows():
                        data_points.append({
                            "Date": date.strftime('%Y-%m-%d'),
                            "Close": round(float(row['Close']), 4)
                        })
                    results.append({
                        "symbol": symbol,
                        "history": data_points
                    })
            except Exception as e:
                logger.warning(
                    f"Failed to fetch history for {symbol} ({period}): {e}")
        return {"history": results}
    except Exception as e:
        logger.error(json.dumps({"message": f"Market history failed: {e}", "status_code": 500}))
        raise HTTPException(
            status_code=500, detail="Failed to fetch market history")

async def get_market_news(current_user: User):
    """
    Fetches the latest geopolitical and financial market news from the NewsData API.
    
    Args:
        current_user (User): The authenticated user making the request.
        
    Returns:
        dict: Categorized news articles (geopolitical and market) with titles and URLs.
        
    Raises:
        HTTPException (500): If the news API request fails.
    """
    try:
        with NewsDataApiClient(apikey=NEWSDATA_API_KEY) as api:
            geo_response = api.latest_api(
                category="politics,world",
                q="NOT (finance OR economy OR market OR stock)",
                language="en"
            )
            geo_results = geo_response.get('results', [])
            market_response = api.market_api(
                qInMeta="finance OR economy OR market",
                language="en"
            )
            market_results = market_response.get('results', [])

            def format_news(results):
                formatted = []
                for r in results[:5]:
                    formatted.append({
                        "title": r.get('title', 'No Title'),
                        "snippet": r.get('description') or r.get('content') or 'No Snippet available.',
                        "url": r.get('link', ''),
                        "source": r.get('source_name', 'Unknown'),
                        "image_url": r.get('image_url')
                    })
                return formatted

            return {
                "geopolitical": format_news(geo_results),
                "market": format_news(market_results)
            }
    except Exception as e:
        logger.error(json.dumps({"message": f"Market news failed: {e}", "status_code": 500}))
        raise HTTPException(status_code=500, detail="Failed to fetch news")
