import json
import logging
import asyncio
import os
import warnings


import langchain_openai.chat_models.base as base

original_convert_delta = base._convert_delta_to_message_chunk

def patched_convert_delta(_dict, default_class):
    chunk = original_convert_delta(_dict, default_class)
    if "reasoning_content" in _dict and _dict["reasoning_content"] is not None:
        chunk.additional_kwargs["reasoning_content"] = _dict["reasoning_content"]
    return chunk

base._convert_delta_to_message_chunk = patched_convert_delta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from config import engine, DATABASE_URL, ALLOWED_ORIGINS, SessionLocal
from middleware.cache_middleware import StripCacheControlMiddleware
from routes.auth_routes import auth_router, callback_router
from routes.account_routes import router as account_router
from routes.chat_routes import router as chat_router
from routes.media_routes import router as media_router
from routes.categorizer_routes import router as categorizer_router
from routes.advisor_routes import router as advisor_router
from routes.market_routes import router as market_router
from routes.memory_routes import router as memory_router
from routes.analytics_routes import router as analytics_router
from services.api_integrator.access_token_generator import AccessTokenGenerator
from services.db_service import init_db
from services.mcp_bridge import MCPBridge
from services.logger_setup import get_core_logger
from models.database_models import Bank

logger = get_core_logger(__name__)

init_db()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info({"message": f"INITIALIZING BUDAI CORE ENGINE", "status_code": 200})
    bridge = MCPBridge()
    FastAPICache.init(InMemoryBackend(), prefix="budai-cache")
    logger.info({"message": f"Background tasks have been offloaded to Prefect workers.", "status_code": 200})
    yield
    logger.info({"message": f"Shutting down core engine", "status_code": 200})


app = FastAPI(title="BudAI API Core", version="2.0.0", lifespan=lifespan)


app.add_middleware(StripCacheControlMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Vercel-AI-Data-Stream"],
)

from routes.webhook_routes import router as webhook_router
from routes.widget_routes import router as widget_router
from routes.intent_routes import router as intent_router
from routes.dashboard_routes import router as dashboard_router
from routes.onboarding_routes import router as onboarding_router
from routes.bucket_routes import router as bucket_router
from routes.websocket_routes import router as websocket_router

app.include_router(auth_router)
app.include_router(callback_router)
app.include_router(account_router)
app.include_router(chat_router)
app.include_router(media_router)
app.include_router(categorizer_router)
app.include_router(advisor_router)
app.include_router(market_router)
app.include_router(webhook_router)
app.include_router(memory_router)
app.include_router(analytics_router)
app.include_router(widget_router)
app.include_router(intent_router)
app.include_router(dashboard_router)
app.include_router(onboarding_router)
app.include_router(bucket_router)
app.include_router(websocket_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
