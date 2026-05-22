from fastapi import APIRouter, Depends, HTTPException
from fastapi_cache.decorator import cache
from middleware.auth_middleware import get_current_user
from models.database_models import User
import yfinance as yf
from newsdataapi import NewsDataApiClient
from config import NEWSDATA_API_KEY
from services.logger_setup import get_core_logger
from utils.cache_utils import global_cache_key_builder

logger = get_core_logger(__name__)

market_router = APIRouter(prefix="/api/market", tags=["market"])


@market_router.get("/ticker")
@cache(expire=900, namespace="market_ticker")
async def get_market_ticker(current_user: User = Depends(get_current_user)):
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
                logger.warning(f"Failed to fetch ticker {symbol}: {e}")
        return {"tickers": results}
    except Exception as e:
        logger.error(f"Market ticker failed: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to fetch market data")


@market_router.get("/history")
@cache(expire=3600, namespace="market_history")
async def get_market_history(range: str = "1M", current_user: User = Depends(get_current_user)):
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
        logger.error(f"Market history failed: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to fetch market history")


@market_router.get("/news")
@cache(expire=7200, namespace="market_news", key_builder=global_cache_key_builder)
async def get_market_news(current_user: User = Depends(get_current_user)):
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
        logger.error(f"Market news failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch news")
