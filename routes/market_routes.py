from fastapi import APIRouter, Depends, Query
from fastapi_cache.decorator import cache
from middleware.auth_middleware import get_current_user
from models.database_models import User
from utils.cache_utils import global_cache_key_builder

from controllers.market.get import get_market_ticker, get_market_history, get_market_news, calculate_fx_rate

router = APIRouter(prefix="/api/market", tags=["market"])

@router.get("/ticker")
@cache(expire=900, namespace="market_ticker")
async def get_market_ticker_route(current_user: User = Depends(get_current_user)):
    return await get_market_ticker(current_user)

@router.get("/history")
@cache(expire=3600, namespace="market_history")
async def get_market_history_route(range: str = "1M", current_user: User = Depends(get_current_user)):
    return await get_market_history(range, current_user)

@router.get("/news")
@cache(expire=7200, namespace="market_news", key_builder=global_cache_key_builder)
async def get_market_news_route(current_user: User = Depends(get_current_user)):
    return await get_market_news(current_user)

@router.get("/fx")
@router.get("/fx-rate")
@cache(expire=300, namespace="market_fx")
async def get_fx_rate_route(
    from_currency: str = Query(..., description="Base 3-letter currency code (e.g. GBP, EUR, USD)", min_length=3, max_length=5),
    to_currency: str = Query(..., description="Target 3-letter currency code (e.g. USD, JPY, EUR)", min_length=3, max_length=5),
    amount: float = Query(1.0, ge=0.0, description="Amount to convert, defaults to 1.0"),
    current_user: User = Depends(get_current_user)
):
    """
    Calculates the live foreign exchange rate between two currencies.
    Attempts direct market pair first; if direct quote is unavailable,
    triangulates the cross-rate using USD as the base reference asset.
    """
    return await calculate_fx_rate(from_currency=from_currency, to_currency=to_currency, amount=amount)
