import logging
import asyncio
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from config import Base, engine, DATABASE_URL, ALLOWED_ORIGINS, SessionLocal
from middleware.cache_middleware import StripCacheControlMiddleware
from controllers.auth_controller import auth_router, callback_router
from controllers.account_controller import account_router
from controllers.chat_controller import chat_router
from controllers.media_controller import media_router
from controllers.categorizer_controller import categorizer_router
from controllers.advisor_controller import advisor_router
from controllers.market_controller import market_router
from services.api_integrator.access_token_generator import AccessTokenGenerator
from services.db_service import init_db
from services.mcp_bridge import MCPBridge
from services.logger_setup import get_core_logger
from models.database_models import Bank

logger = get_core_logger(__name__)

init_db()

def refresh_all_tokens():
    try:
        with SessionLocal() as session:
            providers = session.query(Bank.bank_name, Bank.truelayer_provider_id, Bank.user_uuid).all()
        if not providers:
            return
        logger.info(f"Starting background refresh for {len(providers)} providers.")
        token_gen = AccessTokenGenerator()
        for bank_name, provider_id, user_uuid in providers:
            try:
                success = token_gen.refresh_token(provider_id, user_uuid)
                if success:
                    logger.info(f"Successfully refreshed tokens for {bank_name}.")
                else:
                    logger.warning(f"Failed to refresh tokens for {bank_name}.")
            except Exception as e:
                logger.error(f"Error refreshing {bank_name}: {e}")
    except Exception as e:
        logger.error(f"Critical error in token refresh scheduler: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("INITIALIZING BUDAI CORE ENGINE")
    bridge = MCPBridge()
    logger.info(f"Local Workspace: {bridge.workspace_dir}")
    FastAPICache.init(InMemoryBackend(), prefix="budai-cache")
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=refresh_all_tokens, trigger="interval", minutes=45)
    scheduler.start()
    logger.info("Background scheduler started for token refresh (45m interval)")
    yield
    scheduler.shutdown()
    logger.info("Shutting down background scheduler")

app = FastAPI(title="BudAI API Core", version="2.0.0", lifespan=lifespan)

app.add_middleware(StripCacheControlMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(callback_router)
app.include_router(account_router)
app.include_router(chat_router)
app.include_router(media_router)
app.include_router(categorizer_router)
app.include_router(advisor_router)
app.include_router(market_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
