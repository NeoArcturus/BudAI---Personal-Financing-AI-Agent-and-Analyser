import sqlite3


def init_db(db_path="budai_memory.db"):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_uuid TEXT PRIMARY KEY,
                name TEXT,
                password TEXT,
                user_type TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS banks (
                bank_uuid TEXT PRIMARY KEY,
                user_uuid TEXT REFERENCES users(user_uuid),
                truelayer_provider_id TEXT,
                bank_name TEXT,
                bank_logo_uri TEXT,
                access_token BLOB,
                refresh_token BLOB,
                consent_status TEXT,
                consent_status_updated_at TIMESTAMP,
                consent_created_at TIMESTAMP,
                consent_expires_at TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                account_id TEXT PRIMARY KEY,
                user_uuid TEXT REFERENCES users(user_uuid),
                bank_uuid TEXT REFERENCES banks(bank_uuid),
                account_number TEXT,
                sort_code TEXT,
                account_type TEXT,
                account_balance REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_uuid TEXT PRIMARY KEY,
                user_uuid TEXT REFERENCES users(user_uuid),
                bank_uuid TEXT REFERENCES banks(bank_uuid),
                account_id TEXT REFERENCES accounts(account_id),
                date TIMESTAMP,
                amount REAL,
                category TEXT,
                description TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_uuid TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
