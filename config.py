import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

# --- Database & Security ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///budai_memory.db")
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-jwt-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# --- TrueLayer API Credentials ---
TRUELAYER_BASE_URL = os.getenv("BASE_URL", "https://api.truelayer.com/data/v1")
TRUELAYER_AUTH_URL = os.getenv("AUTH_LINK_URL", "https://auth.truelayer.com")
TRUELAYER_CLIENT_ID = os.getenv("CLIENT_ID")
TRUELAYER_CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TRUELAYER_REDIRECT_URI = os.getenv("REDIRECT_URI")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
ALLOWED_ORIGINS = [
    origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", FRONTEND_URL).split(",") if origin.strip()
]

# --- Encryption Key for TrueLayer Tokens ---
ENCRYPTION_KEY = os.getenv(
    "ENCRYPTION_KEY", b'cw_8H_1M4bX_3nF8vO5n3Y7A8xQ3_1m8aT2vP5_v5r8=')

# --- SQLAlchemy Setup ---
engine = create_engine(
    DATABASE_URL, connect_args={
        "check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
