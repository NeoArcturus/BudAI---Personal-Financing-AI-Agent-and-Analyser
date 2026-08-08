from fastapi import APIRouter, Depends
from fastapi_cache.decorator import cache
from middleware.auth_middleware import get_current_user
from models.database_models import User
from utils.cache_utils import global_cache_key_builder

from controllers.market.get import get_market_ticker, get_market_history, get_market_news

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
