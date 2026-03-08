from flask import Flask
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import sqlite3
import os

from services.db_service import init_db
from routes.auth_routes import auth_bp, callback_bp  # IMPORT THE NEW BLUEPRINT
from routes.account_routes import account_bp
from routes.chat_routes import chat_bp
from routes.media_routes import media_bp
from services.api_integrator.access_token_generator import AccessTokenGenerator

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Pre-fixed API routes
app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(account_bp, url_prefix='/api/accounts')
app.register_blueprint(chat_bp, url_prefix='/api/chat')
app.register_blueprint(media_bp, url_prefix='/api/media')

# NEW: Register the callback at the root level (no prefix)
app.register_blueprint(callback_bp)


def refresh_all_tokens():
    try:
        with sqlite3.connect("budai_memory.db") as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT truelayer_account_id, user_uuid FROM accounts WHERE access_status = 'active'")
            providers = cursor.fetchall()

        if not providers:
            return

        token_gen = AccessTokenGenerator()
        for provider_id, user_uuid in providers:
            token_gen.refresh_token(provider_id, user_uuid)
    except Exception:
        pass


if __name__ == '__main__':
    init_db()

    scheduler = BackgroundScheduler()
    scheduler.add_job(func=refresh_all_tokens, trigger="interval", minutes=45)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())

    app.run(port=8080, threaded=True, debug=False)
