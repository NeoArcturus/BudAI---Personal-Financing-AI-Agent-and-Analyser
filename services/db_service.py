import json
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from urllib.parse import urlparse
from config import engine, DATABASE_URL
from sqlmodel import SQLModel
from services.logger_setup import get_core_logger

logger = get_core_logger("db_service")

def ensure_db_exists():
    if "postgresql" not in DATABASE_URL:
        return
        
    logger.debug(json.dumps({"message": f"Ensuring PostgreSQL database exists", "status_code": 100}))
    result = urlparse(DATABASE_URL)
    username = result.username
    password = result.password
    database = result.path.lstrip('/')
    hostname = result.hostname
    port = result.port or 5432
    
    try:
        conn = psycopg2.connect(
            dbname='postgres',
            user=username,
            password=password,
            host=hostname,
            port=port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{database}'")
        exists = cur.fetchone()
        
        if not exists:
            logger.info(json.dumps({"message": f"Database {database} does not exist. Creating it.", "status_code": 200}))
            cur.execute(f'CREATE DATABASE {database}')
            logger.info(json.dumps({"message": f"Database {database} created successfully.", "status_code": 200}))
        else:
            logger.debug(json.dumps({"message": f"Database {database} already exists.", "status_code": 100}))
            
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(json.dumps({"message": f"Error ensuring database exists: {e}", "status_code": 500}))

def init_db(db_path=None):
    try:
        ensure_db_exists()
        
        from sqlalchemy import text
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
        SQLModel.metadata.create_all(bind=engine)
        
        from sqlalchemy import text
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS sub_category VARCHAR;"))
            conn.execute(text("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS is_semantic_anomaly BOOLEAN DEFAULT FALSE;"))
            conn.execute(text("ALTER TABLE subscriptions ADD COLUMN IF NOT EXISTS cluster_signature_hash VARCHAR UNIQUE;"))

        logger.debug(json.dumps({"message": f"Database initialized successfully", "status_code": 100}))
    except Exception as e:
        logger.error(json.dumps({"message": f"Failed to initialize database: {e}", "status_code": 500}))
        raise
