import sqlite3


def init_db(db_path="budai_memory.db"):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_uuid TEXT PRIMARY KEY,
                name TEXT,
                password TEXT,
                truelayer_account_ids TEXT,
                user_type TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_uuid TEXT PRIMARY KEY,
                user_uuid TEXT,
                truelayer_account_id TEXT,
                csv_file_path TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                user_uuid TEXT,
                truelayer_account_id TEXT PRIMARY KEY,
                bank_name TEXT,
                sort_code TEXT,
                account_number TEXT,
                account_balance REAL,
                account_label TEXT,
                access_token BLOB,
                refresh_token BLOB,
                access_token_validity_time TIMESTAMP,
                connection_extension_status TEXT,
                access_status TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auth_states (
                state_uuid TEXT PRIMARY KEY,
                user_uuid TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
