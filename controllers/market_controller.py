import json
from fastapi import HTTPException
from models.database_models import User
import yfinance as yf
from newsdataapi import NewsDataApiClient
from config import NEWSDATA_API_KEY
from services.logger_setup import get_core_logger
from services.analytics.fx_service import get_cross_fx_rate

logger = get_core_logger(__name__)


async def get_market_ticker(current_user: User):
    """Fetches real-time price and daily change percentage for assets."""
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
        raise HTTPException(status_code=500, detail="Failed to fetch market data")


async def get_market_history(range: str, current_user: User):
    """Retrieves historical prices for predefined market assets."""
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
                logger.warning(f"Failed to fetch history for {symbol} ({period}): {e}")
        return {"history": results}
    except Exception as e:
        logger.error(json.dumps({"message": f"Market history failed: {e}", "status_code": 500}))
        raise HTTPException(status_code=500, detail="Failed to fetch market history")


async def get_market_news(current_user: User):
    """Fetches market news."""
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


async def calculate_fx_rate(from_currency: str, to_currency: str, amount: float = 1.0) -> dict:
    """Calculates live FX conversion rate."""
    try:
        res = get_cross_fx_rate(from_currency=from_currency, to_currency=to_currency, amount=amount)
        return {"status": "success", "data": res}
    except ValueError as ve:
        logger.warning(json.dumps({"message": f"FX rate calculation error: {ve}", "status_code": 400}))
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(json.dumps({"message": f"Unexpected FX calculation failure: {e}", "status_code": 500}))
        raise HTTPException(status_code=500, detail=f"Failed to calculate exchange rate: {str(e)}")
