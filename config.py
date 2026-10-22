import os
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
from sqlmodel import create_engine
from services.logger_setup import get_core_logger
import redis

logger = get_core_logger(__name__)

load_dotenv()

# Automatically map Prefect Cloud variables if defined with CLOUD suffix
if os.getenv("PREFECT_CLOUD_API_URL") and not os.getenv("PREFECT_API_URL"):
    os.environ["PREFECT_API_URL"] = os.getenv("PREFECT_CLOUD_API_URL")
if os.getenv("PREFECT_CLOUD_API_KEY") and not os.getenv("PREFECT_API_KEY"):
    os.environ["PREFECT_API_KEY"] = os.getenv("PREFECT_CLOUD_API_KEY")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/budai")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

DATABASE_URL_UNPOOLED = os.getenv("DATABASE_URL_UNPOOLED", DATABASE_URL)
if DATABASE_URL_UNPOOLED.startswith("postgres://"):
    DATABASE_URL_UNPOOLED = DATABASE_URL_UNPOOLED.replace("postgres://", "postgresql://", 1)

if os.path.exists("/.dockerenv") and ("localhost" in DATABASE_URL or "127.0.0.1" in DATABASE_URL):
    DATABASE_URL = DATABASE_URL.replace("localhost", "budai-db").replace("127.0.0.1", "budai-db")

# Neon Platform Services configuration
NEON_BRANCH = os.getenv("NEON_BRANCH", "production")
NEON_AUTH_BASE_URL = os.getenv("NEON_AUTH_BASE_URL")
NEON_AUTH_JWKS_URL = os.getenv("NEON_AUTH_JWKS_URL")
NEON_FUNCTION_HELLO_BASE_URL = os.getenv("NEON_FUNCTION_HELLO_BASE_URL")

# Neon Object Storage (S3-compatible)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_ENDPOINT_URL_S3 = os.getenv("AWS_ENDPOINT_URL_S3")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
NEON_STORAGE_BUCKET = os.getenv("NEON_STORAGE_BUCKET", "assets")

# Qdrant Vector Cloud
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

import secrets
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
TRUELAYER_BASE_URL = os.getenv("BASE_URL", "https://api.truelayer.com/data/v1")
TRUELAYER_AUTH_URL = os.getenv("AUTH_LINK_URL", "https://auth.truelayer.com")
TRUELAYER_CLIENT_ID = os.getenv("CLIENT_ID")
TRUELAYER_CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TRUELAYER_REDIRECT_URI = os.getenv("REDIRECT_URI")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000").strip("\"\'")
raw_origins = os.getenv("ALLOWED_ORIGINS", FRONTEND_URL)
if "bud-ai-platform.vercel.app" not in raw_origins:
    raw_origins = f"{raw_origins},https://bud-ai-platform.vercel.app"
ALLOWED_ORIGINS = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", b'cw_8H_1M4bX_3nF8vO5n3Y7A8xQ3_1m8aT2vP5_v5r8=')

# Pooled engine for normal application query traffic (PgBouncer)
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_timeout=60,
    pool_recycle=1800,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Direct (unpooled) engine for migrations, DDL, and session-level operations
direct_engine = create_engine(
    DATABASE_URL_UNPOOLED,
    pool_size=5,
    max_overflow=10,
    pool_timeout=60,
    pool_recycle=1800,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL_UNPOOLED else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
