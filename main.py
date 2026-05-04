import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import sqlite3
from contextlib import asynccontextmanager
from config import Base, engine, DATABASE_URL, ALLOWED_ORIGINS
from controllers.auth_controller import auth_router, callback_router
from controllers.account_controller import account_router
from controllers.chat_controller import chat_router
from controllers.media_controller import media_router
from controllers.categorizer_controller import categorizer_router
from services.api_integrator.access_token_generator import AccessTokenGenerator
from services.mcp_bridge import MCPBridge

logger = logging.getLogger("uvicorn.error")

Base.metadata.create_all(bind=engine)


def refresh_all_tokens():
    try:
        db_path = DATABASE_URL.replace("sqlite:///", "")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT truelayer_provider_id, user_uuid FROM banks")
            providers = cursor.fetchall()
        if not providers:
            return
        token_gen = AccessTokenGenerator()
        for provider_id, user_uuid in providers:
            token_gen.refresh_token(provider_id, user_uuid)
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("="*50)
    logger.info("INITIALIZING BUDAI CORE ENGINE")

    bridge = MCPBridge()
    logger.info(f"Local Workspace: {bridge.workspace_dir}")
    logger.info("Inbound Filesystem Watchdog: DISABLED")

    logger.info("="*50)

    scheduler = BackgroundScheduler()
    scheduler.add_job(func=refresh_all_tokens, trigger="interval", minutes=45)
    scheduler.start()

    yield

    scheduler.shutdown()

app = FastAPI(title="BudAI API Core", version="2.0.0", lifespan=lifespan)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
