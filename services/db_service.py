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
            
        # This registers the models
        from models.database_models import User, Bank, Account, Transaction, ChatSession, ChatHistory, UserLifestyleProfile, LifestyleCluster, Subscription, ProactiveInsight, MerchantKnowledge, Bucket, VirtualTransfer, SystemAlert
        SQLModel.metadata.create_all(bind=engine)
        
        from sqlalchemy import text
        with engine.begin() as conn:
            # Legacy migrations
            conn.execute(text("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS sub_category VARCHAR;"))
            conn.execute(text("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS is_semantic_anomaly BOOLEAN DEFAULT FALSE;"))
            conn.execute(text("ALTER TABLE subscriptions ADD COLUMN IF NOT EXISTS cluster_signature_hash VARCHAR UNIQUE;"))

            # PHASE 1: Agentic Database Architecture
            # 1. Trigger for cached_balance updates
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION update_bucket_balances()
                RETURNS TRIGGER AS $$
                BEGIN
                    -- Deduct from source bucket
                    UPDATE buckets
                    SET cached_balance = cached_balance - NEW.amount
                    WHERE id = NEW.source_bucket_id;

                    -- Add to target bucket
                    UPDATE buckets
                    SET cached_balance = cached_balance + NEW.amount
                    WHERE id = NEW.target_bucket_id;

                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """))

            # Create or replace the trigger
            conn.execute(text("""
                DROP TRIGGER IF EXISTS trigger_update_bucket_balances ON virtual_transfers;
                CREATE TRIGGER trigger_update_bucket_balances
                AFTER INSERT ON virtual_transfers
                FOR EACH ROW
                EXECUTE FUNCTION update_bucket_balances();
            """))

            # 2. CHECK constraint: non-DEFAULT buckets cannot drop below 0
            conn.execute(text("""
                ALTER TABLE buckets 
                DROP CONSTRAINT IF EXISTS check_non_default_balance_positive;
            """))
            conn.execute(text("""
                ALTER TABLE buckets 
                ADD CONSTRAINT check_non_default_balance_positive 
                CHECK (
                    type = 'DEFAULT' OR cached_balance >= 0
                );
            """))

        logger.debug(json.dumps({"message": f"Database initialized successfully", "status_code": 100}))
        
        # Phase 2 prep: Ensure all users have a DEFAULT bucket
        seed_default_buckets()
        
    except Exception as e:
        logger.error(json.dumps({"message": f"Failed to initialize database: {e}", "status_code": 500}))
        raise

def seed_default_buckets():
    import uuid
    from config import SessionLocal
    from sqlalchemy import select
    from models.database_models import User, Bucket
    
    try:
        with SessionLocal() as session:
            users = session.execute(select(User)).scalars().all()
            for user in users:
                default_bucket = session.execute(
                    select(Bucket)
                    .where(Bucket.user_id == user.user_uuid)
                    .where(Bucket.type == "DEFAULT")
                ).scalars().first()
                
                if not default_bucket:
                    logger.info(json.dumps({"message": f"Seeding missing DEFAULT bucket for user {user.user_uuid}", "status_code": 200}))
                    new_bucket = Bucket(
                        id=str(uuid.uuid4()),
                        user_id=user.user_uuid,
                        type="DEFAULT",
                        name="Unallocated Funds",
                        priority_index=100.0,
                        cached_balance=0.0
                    )
                    session.add(new_bucket)
            session.commit()
    except Exception as e:
        logger.error(json.dumps({"message": f"Error seeding default buckets: {e}", "status_code": 500}))
