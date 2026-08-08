import logging
import asyncio
import os
import warnings

warnings.filterwarnings("ignore", message=".*extra_body.*")

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
from apscheduler.schedulers.background import BackgroundScheduler
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

def refresh_all_tokens():
    try:
        with SessionLocal() as session:
            providers = session.query(Bank.bank_name, Bank.truelayer_provider_id, Bank.user_uuid).all()
        if not providers:
            return
        token_gen = AccessTokenGenerator()
        for bank_name, provider_id, user_uuid in providers:
            try:
                success = token_gen.refresh_token(provider_id, user_uuid)
                if success:
                    logger.debug(f"Successfully refreshed tokens for {bank_name}.")
                else:
                    logger.warning(f"Failed to refresh tokens for {bank_name}.")
            except Exception as e:
                logger.error(f"Error refreshing {bank_name}: {e}")
    except Exception as e:
        logger.error(f"Critical error in token refresh scheduler: {e}")

def run_global_lifestyle_analytics():
    try:
        from models.database_models import User
        from services.analytics.lifestyle_clustering import LifestyleClusteringService
        from services.analytics.subscription_detector import SubscriptionDetector
        logger.info("Starting global background job: Analytics & Subscriptions")
        
        with SessionLocal() as session:
            users = session.query(User).all()
            
        if not users:
            return
            
        cluster_service = LifestyleClusteringService()
        sub_detector = SubscriptionDetector()
        
        for user in users:
            try:
                cluster_service.analyze_user_lifestyle(user.user_uuid)
                sub_detector.analyze_user_subscriptions(user.user_uuid)
            except Exception as e:
                logger.error(f"Failed analytics for user {user.user_uuid}: {e}")
                
        logger.info("Global background job completed: Analytics & Subscriptions")
    except Exception as e:
        logger.error(f"Critical error in lifestyle analytics scheduler: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("INITIALIZING BUDAI CORE ENGINE")
    bridge = MCPBridge()
    FastAPICache.init(InMemoryBackend(), prefix="budai-cache")
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=refresh_all_tokens, trigger="interval", minutes=45)
    
    # Run HDBSCAN analytics every 10 minutes for testing
    scheduler.add_job(func=run_global_lifestyle_analytics, trigger="interval", minutes=10)
    
    scheduler.start()
    logger.info("Background scheduler started: Token Refresh (45m) & Lifestyle Analytics (10m)")
    def _run_global_training():
        try:
            from services.memory_service import MemoryService
            logger.info("Pre-warming local ML embedding model...")
            MemoryService() # Initialize singleton to pre-load embedding model
            logger.info("ML embedding model initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize MemoryService: {e}")
            
        try:
            from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
            agent = CategorizerAgent()
            res = agent.train_global()
            if res.get("trained"):
                logger.debug(f"Global categorization model trained on {res.get('samples', 0)} samples.")
            else:
                logger.debug(f"Global categorization model initialization: {res.get('reason')}")
        except Exception as e:
            logger.error(f"Failed to initialize global categorizer: {e}")
            
    asyncio.create_task(asyncio.to_thread(_run_global_training))
    
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
    expose_headers=["X-Vercel-AI-Data-Stream"],
)

from routes.webhook_routes import router as webhook_router

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
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
