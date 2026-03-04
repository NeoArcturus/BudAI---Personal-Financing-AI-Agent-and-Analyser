from flask import Flask, request, jsonify, redirect, send_from_directory
from flask_cors import CORS
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime, timezone
import os
import pandas as pd
import sqlite3
import requests
import traceback
from BudAI_chat import agent_executor
from api_integrator.access_token_generator import AccessTokenGenerator
from api_integrator.get_account_detail import UserAccount

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def init_credentials_db(db_path="budai_memory.db"):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS credentials (
                provider_id TEXT PRIMARY KEY,
                provider_name TEXT,
                access_token TEXT,
                refresh_token TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    return jsonify({"status": "success", "user": data.get("email")})


@app.route('/api/truelayer/status', methods=['GET'])
def truelayer_status():
    token_gen = AccessTokenGenerator()
    return jsonify({"auth_url": token_gen.get_auth_link()})


@app.route('/callback', methods=['GET'])
def truelayer_callback():
    code = request.args.get('code')
    if code:
        token_gen = AccessTokenGenerator()
        if token_gen.generate_token_from_code(code):
            return redirect("http://localhost:3000/dashboard")
    return jsonify({"error": "Authentication failed"}), 400


@app.route('/api/accounts', methods=['GET'])
def get_accounts():
    all_accounts = []
    try:
        with sqlite3.connect("budai_memory.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT provider_id, access_token FROM credentials")
            rows = cursor.fetchall()
        for provider_id, token in rows:
            headers = {"Authorization": f"Bearer {token}",
                       "accept": "application/json"}
            acc_resp = requests.get(
                "https://api.truelayer.com/data/v1/accounts", headers=headers)
            if acc_resp.status_code == 200:
                for acc in acc_resp.json().get("results", []):
                    acc_id = acc.get("account_id")
                    bal_res = requests.get(
                        f"https://api.truelayer.com/data/v1/accounts/{acc_id}/balance", headers=headers).json()
                    balance = bal_res.get("results", [{}])[
                        0] if bal_res.get("results") else {}
                    all_accounts.append({
                        "account_id": acc_id,
                        "provider_name": acc.get("provider", {}).get("display_name", "Unknown Bank"),
                        "account_number": acc.get("account_number", {}).get("number", "")[-4:],
                        "sort_code": acc.get("account_number", {}).get("sort_code", ""),
                        "currency": balance.get("currency", "GBP"),
                        "balance": balance.get("available", balance.get("current", 0.0))
                    })
        return jsonify({"accounts": all_accounts})
    except Exception:
        return jsonify({"accounts": []}), 500


@app.route('/api/accounts/<account_id>/transactions', methods=['GET'])
def get_transactions(account_id):
    try:
        with sqlite3.connect("budai_memory.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT access_token FROM credentials")
            tokens = [row[0] for row in cursor.fetchall()]
        df = None
        for token in tokens:
            user_acc = UserAccount()
            user_acc.access_token = token
            user_acc.headers = {
                "Authorization": f"Bearer {token}", "accept": "application/json"}
            user_acc.account_id = account_id
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            temp_df = user_acc.all_transactions()
            if temp_df is not None and not temp_df.empty:
                df = temp_df
                break
        if df is not None and not df.empty:
            return jsonify({"transactions": df.fillna("").to_dict(orient="records")})
        return jsonify({"transactions": []})
    except Exception:
        traceback.print_exc()
        return jsonify({"transactions": []}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('input')
    raw_history = data.get('chat_history', [])
    chat_history = []
    for msg in raw_history:
        if msg.get('role') == 'user':
            chat_history.append(HumanMessage(content=msg.get('text')))
        else:
            chat_history.append(AIMessage(content=msg.get('text')))
    try:
        response = agent_executor.invoke(
            {"input": user_input, "chat_history": chat_history})
        return jsonify({"output": response["output"]})
    except Exception:
        return jsonify({"output": "Internal engine error."}), 500


@app.route('/api/media/csv/<filename>')
def serve_csv(filename):
    csv_dir = os.path.join(os.getcwd(), 'saved_media', 'csvs')
    try:
        df = pd.read_csv(os.path.join(csv_dir, filename))
        return jsonify({"data": df.fillna("").to_dict(orient="records")})
    except Exception:
        return jsonify({"data": []}), 404


@app.route('/api/media/image/<filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'saved_media', 'images'), filename)


if __name__ == '__main__':
    init_credentials_db()
    app.run(port=8080, threaded=True, debug=False)
