from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import sqlite3
import atexit
from config import Base, engine, DATABASE_URL, ALLOWED_ORIGINS

from controllers.auth_controller import auth_router, callback_router
from controllers.account_controller import account_router
from controllers.chat_controller import chat_router
from controllers.media_controller import media_router
from controllers.categorizer_controller import categorizer_router
from services.api_integrator.access_token_generator import AccessTokenGenerator

Base.metadata.create_all(bind=engine)

app = FastAPI(title="BudAI API Core", version="2.0.0")

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
    except Exception:
        pass


@app.on_event("startup")
def startup_event():
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=refresh_all_tokens, trigger="interval", minutes=45)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
