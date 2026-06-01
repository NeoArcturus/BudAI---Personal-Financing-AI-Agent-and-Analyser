import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from services.logger_setup import get_core_logger
logger = get_core_logger(__name__)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/budai")

if os.path.exists("/.dockerenv") and ("localhost" in DATABASE_URL or "127.0.0.1" in DATABASE_URL):
    DATABASE_URL = DATABASE_URL.replace("localhost", "budai-db").replace("127.0.0.1", "budai-db")

SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-jwt-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
TRUELAYER_BASE_URL = os.getenv("BASE_URL", "https://api.truelayer.com/data/v1")
TRUELAYER_AUTH_URL = os.getenv("AUTH_LINK_URL", "https://auth.truelayer.com")
TRUELAYER_CLIENT_ID = os.getenv("CLIENT_ID")
TRUELAYER_CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TRUELAYER_REDIRECT_URI = os.getenv("REDIRECT_URI")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
raw_origins = os.getenv("ALLOWED_ORIGINS")
if not raw_origins:
    raw_origins = FRONTEND_URL
ALLOWED_ORIGINS = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", b'cw_8H_1M4bX_3nF8vO5n3Y7A8xQ3_1m8aT2vP5_v5r8=')

engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_timeout=60,
    pool_recycle=1800,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
