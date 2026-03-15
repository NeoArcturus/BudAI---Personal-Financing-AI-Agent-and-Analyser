from flask import Flask
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import sqlite3
import os

from services.db_service import init_db
from routes.auth_routes import auth_bp, callback_bp
from routes.account_routes import account_bp
from routes.chat_routes import chat_bp
from routes.media_routes import media_bp
from services.api_integrator.access_token_generator import AccessTokenGenerator

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(account_bp, url_prefix='/api/accounts')
app.register_blueprint(chat_bp, url_prefix='/api/chat')
app.register_blueprint(media_bp, url_prefix='/api/media')

app.register_blueprint(callback_bp)


def refresh_all_tokens():
    try:
        with sqlite3.connect("budai_memory.db") as conn:
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


def display_all_db_data():
    try:
        db_path = "budai_memory.db"
        if not os.path.exists(db_path):
            return

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            print("\n" + "╔" + "═"*78 + "╗")
            print("║" + " BUDAI SYSTEM DATABASE INSPECTION ".center(78) + "║")
            print("╚" + "═"*78 + "╝")

            for table_name in tables:
                name = table_name[0]
                if name == 'sqlite_sequence':
                    continue

                cursor.execute(f"SELECT * FROM {name}")
                columns = [description[0]
                           for description in cursor.description]
                rows = cursor.fetchall()

                print(f"\nTABLE: {name.upper()}")

                if not rows:
                    print("   [Empty Table]")
                    continue

                col_widths = [max(len(str(col)), 10) for col in columns]
                for row in rows:
                    for i, val in enumerate(row):
                        col_widths[i] = max(col_widths[i], len(str(val)))

                header = " | ".join(str(columns[i]).ljust(
                    col_widths[i]) for i in range(len(columns)))
                separator = "-+-".join("-" * col_widths[i]
                                       for i in range(len(columns)))

                print(header)
                print(separator)

                for row in rows:
                    row_str = " | ".join(str(row[i]).ljust(
                        col_widths[i]) for i in range(len(row)))
                    print(row_str)

            print("\n" + "="*80 + "\n")
    except Exception as e:
        print(f"Database inspection failed: {e}")


if __name__ == '__main__':
    init_db()

    scheduler = BackgroundScheduler()
    scheduler.add_job(func=refresh_all_tokens, trigger="interval", minutes=45)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())

    app.run(port=8080, threaded=True, debug=False)
