import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from urllib.parse import urlparse
from config import Base, engine, DATABASE_URL
from services.logger_setup import get_core_logger

logger = get_core_logger("db_service")

def ensure_db_exists():
    if "postgresql" not in DATABASE_URL:
        return
        
    logger.info("Ensuring PostgreSQL database exists")
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
            logger.info(f"Database {database} does not exist. Creating it.")
            cur.execute(f'CREATE DATABASE {database}')
            logger.info(f"Database {database} created successfully.")
        else:
            logger.info(f"Database {database} already exists.")
            
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error ensuring database exists: {e}")

def init_db(db_path=None):
    try:
        ensure_db_exists()
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
