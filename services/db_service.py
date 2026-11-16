import json
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from urllib.parse import urlparse
from config import engine, direct_engine, DATABASE_URL
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
    
    # If using managed cloud Postgres (Neon/Supabase/Render), the database is pre-created in the console.
    if "neon.tech" in DATABASE_URL or "sslmode=require" in DATABASE_URL:
        logger.debug(json.dumps({"message": "Using managed cloud PostgreSQL, skipping raw database creation.", "status_code": 100}))
        return

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
        logger.warning(json.dumps({"message": f"Database check skipped or failed: {e}", "status_code": 200}))

def init_db(db_path=None):
    try:
        ensure_db_exists()
        
        from sqlalchemy import text
        with direct_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            for val in ["UNALLOCATED", "BILLS", "GOALS"]:
                try:
                    conn.execute(text(f"ALTER TYPE buckettype ADD VALUE IF NOT EXISTS '{val}';"))
                except Exception:
                    pass

        # This registers the models
        from models.database_models import User, Bank, Account, Transaction, ChatSession, ChatHistory, UserLifestyleProfile, LifestyleCluster, Subscription, ProactiveInsight, MerchantKnowledge, Bucket, VirtualTransfer, SystemAlert
        SQLModel.metadata.create_all(bind=direct_engine)
        
        from sqlalchemy import text
        with direct_engine.begin() as conn:
            # Column migrations
            conn.execute(text("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS sub_category VARCHAR;"))
            conn.execute(text("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS is_semantic_anomaly BOOLEAN DEFAULT FALSE;"))
            conn.execute(text("ALTER TABLE subscriptions ADD COLUMN IF NOT EXISTS cluster_signature_hash VARCHAR UNIQUE;"))

            # Buckets table migrations
            conn.execute(text("ALTER TABLE buckets ADD COLUMN IF NOT EXISTS priority_index DOUBLE PRECISION DEFAULT 10.0;"))
            conn.execute(text("ALTER TABLE buckets ALTER COLUMN priority_index DROP NOT NULL;"))
            conn.execute(text("ALTER TABLE buckets ALTER COLUMN priority_index SET DEFAULT 10.0;"))
            conn.execute(text("ALTER TABLE buckets ADD COLUMN IF NOT EXISTS priority INTEGER DEFAULT 10;"))
            conn.execute(text("ALTER TABLE buckets ADD COLUMN IF NOT EXISTS cached_balances JSONB DEFAULT '{}'::jsonb;"))
            conn.execute(text("ALTER TABLE buckets ADD COLUMN IF NOT EXISTS currency VARCHAR DEFAULT 'GBP';"))
            conn.execute(text("ALTER TABLE buckets ADD COLUMN IF NOT EXISTS allocated_amount NUMERIC(15, 2) DEFAULT 0.00;"))
            conn.execute(text("ALTER TABLE buckets ADD COLUMN IF NOT EXISTS status_text VARCHAR;"))
            conn.execute(text("ALTER TABLE buckets ADD COLUMN IF NOT EXISTS has_shortfall_warning BOOLEAN DEFAULT FALSE;"))
            conn.execute(text("ALTER TABLE buckets ADD COLUMN IF NOT EXISTS shortfall_amount NUMERIC(15, 2) DEFAULT 0.00;"))
            conn.execute(text("ALTER TABLE buckets ADD COLUMN IF NOT EXISTS associated_account_id VARCHAR;"))
            conn.execute(text("ALTER TABLE buckets ADD COLUMN IF NOT EXISTS associated_account_name VARCHAR;"))
            conn.execute(text("ALTER TABLE buckets ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;"))
            conn.execute(text("ALTER TABLE buckets ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();"))

            # Update legacy DEFAULT enum/string rows to UNALLOCATED
            conn.execute(text("UPDATE buckets SET type = 'UNALLOCATED' WHERE type = 'DEFAULT';"))

            # 1. Trigger for cached_balance updates
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION update_bucket_balances()
                RETURNS TRIGGER AS $$
                BEGIN
                    UPDATE buckets
                    SET cached_balance = cached_balance - NEW.amount
                    WHERE id = NEW.source_bucket_id;

                    UPDATE buckets
                    SET cached_balance = cached_balance + NEW.amount
                    WHERE id = NEW.target_bucket_id;

                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """))

            conn.execute(text("""
                DROP TRIGGER IF EXISTS trigger_update_bucket_balances ON virtual_transfers;
                CREATE TRIGGER trigger_update_bucket_balances
                AFTER INSERT ON virtual_transfers
                FOR EACH ROW
                EXECUTE FUNCTION update_bucket_balances();
            """))

            # 2. CHECK constraint: non-UNALLOCATED buckets cannot drop below 0
            conn.execute(text("""
                ALTER TABLE buckets 
                DROP CONSTRAINT IF EXISTS check_non_default_balance_positive;
            """))
            conn.execute(text("""
                ALTER TABLE buckets 
                ADD CONSTRAINT check_non_default_balance_positive 
                CHECK (
                    type = 'UNALLOCATED' OR cached_balance >= 0
                );
            """))

            # 3. Triggers for neon_auth."user" synchronization into public.users
            try:
                conn.execute(text("""
                    CREATE OR REPLACE FUNCTION public.sync_neon_auth_user()
                    RETURNS TRIGGER 
                    LANGUAGE plpgsql
                    SECURITY DEFINER
                    SET search_path = public
                    AS $$
                    BEGIN
                        INSERT INTO public.users (user_uuid, email, name, is_onboarded)
                        VALUES (NEW.id::text, NEW.email, NEW.name, FALSE)
                        ON CONFLICT (user_uuid) DO UPDATE 
                        SET email = EXCLUDED.email,
                            name = COALESCE(public.users.name, EXCLUDED.name);
                        RETURN NEW;
                    END;
                    $$;

                    DROP TRIGGER IF EXISTS trg_sync_neon_auth_user ON neon_auth."user";
                    CREATE TRIGGER trg_sync_neon_auth_user
                    AFTER INSERT OR UPDATE ON neon_auth."user"
                    FOR EACH ROW
                    EXECUTE FUNCTION public.sync_neon_auth_user();

                    CREATE OR REPLACE FUNCTION public.delete_neon_auth_user()
                    RETURNS TRIGGER 
                    LANGUAGE plpgsql
                    SECURITY DEFINER
                    SET search_path = public
                    AS $$
                    BEGIN
                        DELETE FROM public.users WHERE user_uuid = OLD.id::text;
                        RETURN OLD;
                    END;
                    $$;

                    DROP TRIGGER IF EXISTS trg_delete_neon_auth_user ON neon_auth."user";
                    CREATE TRIGGER trg_delete_neon_auth_user
                    AFTER DELETE ON neon_auth."user"
                    FOR EACH ROW
                    EXECUTE FUNCTION public.delete_neon_auth_user();

                    GRANT USAGE ON SCHEMA public TO neon_auth;
                    GRANT SELECT, INSERT, UPDATE, DELETE ON public.users TO neon_auth;
                """))
            except Exception as trigger_err:
                logger.debug(f"Neon Auth user sync triggers skipped (neon_auth schema not present): {trigger_err}")

        logger.debug(json.dumps({"message": f"Database initialized successfully", "status_code": 100}))
        
        # Ensure all users have an UNALLOCATED bucket
        seed_default_buckets()
        
    except Exception as e:
        logger.error(json.dumps({"message": f"Failed to initialize database: {e}", "status_code": 500}))
        raise

def seed_default_buckets():
    import uuid
    from config import SessionLocal
    from sqlmodel import select
    from models.database_models import User, Bucket, BucketType
    
    try:
        with SessionLocal() as session:
            users = session.execute(select(User)).scalars().all()
            for user in users:
                unallocated_bucket = session.execute(
                    select(Bucket)
                    .where(Bucket.user_id == user.user_uuid)
                    .where(Bucket.type == BucketType.UNALLOCATED)
                ).scalars().first()
                
                if not unallocated_bucket:
                    logger.info(json.dumps({"message": f"Seeding missing UNALLOCATED bucket for user {user.user_uuid}", "status_code": 200}))
                    new_bucket = Bucket(
                        id=str(uuid.uuid4()),
                        user_id=user.user_uuid,
                        type=BucketType.UNALLOCATED,
                        name="Unallocated",
                        priority=100,
                        cached_balance=0.0,
                        cached_balances={},
                        currency="GBP",
                        is_active=True
                    )
                    session.add(new_bucket)
            session.commit()
    except Exception as e:
        logger.error(json.dumps({"message": f"Error seeding default buckets: {e}", "status_code": 500}))
