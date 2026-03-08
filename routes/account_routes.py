from flask import Blueprint, jsonify, request
import sqlite3
import requests
import traceback
from middleware.auth_middleware import token_required
from cryptography.fernet import Fernet
import os
from services.api_integrator.get_account_detail import UserAccount

account_bp = Blueprint('accounts', __name__)
enc_key = os.getenv(
    "ENCRYPTION_KEY", b'cw_8H_1M4bX_3nF8vO5n3Y7A8xQ3_1m8aT2vP5_v5r8=')
cipher_suite = Fernet(enc_key)


@account_bp.route('/', methods=['GET'])
@token_required
def get_accounts():
    all_accounts = []
    try:
        with sqlite3.connect("budai_memory.db") as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT truelayer_account_id, access_token, access_status, bank_name FROM accounts WHERE user_uuid = ?", (request.user_uuid,))
            rows = cursor.fetchall()

        for provider_id, enc_token, status, bank_name in rows:
            if status == 'revoked':
                all_accounts.append({
                    "account_id": provider_id,
                    "provider_name": bank_name,
                    "account_number": "****",
                    "sort_code": "00-00-00",
                    "currency": "GBP",
                    "balance": 0.0,
                    "status": "revoked",
                    "provider_id": provider_id
                })
                continue

            token = cipher_suite.decrypt(enc_token).decode()
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
                        "provider_name": acc.get("provider", {}).get("display_name", bank_name),
                        "account_number": acc.get("account_number", {}).get("number", "")[-4:],
                        "sort_code": acc.get("account_number", {}).get("sort_code", ""),
                        "currency": balance.get("currency", "GBP"),
                        "balance": balance.get("available", balance.get("current", 0.0)),
                        "status": "active"
                    })
        return jsonify({"accounts": all_accounts})
    except Exception:
        traceback.print_exc()
        return jsonify({"accounts": []}), 500


@account_bp.route('/<account_id>/transactions', methods=['GET'])
@token_required
def get_transactions(account_id):
    try:
        with sqlite3.connect("budai_memory.db") as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT access_token FROM accounts WHERE user_uuid = ? AND access_status = 'active'", (request.user_uuid,))
            enc_tokens = [row[0] for row in cursor.fetchall()]

        df = None
        for enc_token in enc_tokens:
            token = cipher_suite.decrypt(enc_token).decode()
            user_acc = UserAccount(user_id=request.user_uuid)
            user_acc.access_token = token
            user_acc.headers = {
                "Authorization": f"Bearer {token}", "accept": "application/json"}
            user_acc.account_id = account_id
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


@account_bp.route('/<provider_id>', methods=['DELETE'])
@token_required
def revoke_connection(provider_id):
    try:
        with sqlite3.connect("budai_memory.db") as conn:
            conn.execute(
                "DELETE FROM accounts WHERE truelayer_account_id = ? AND user_uuid = ?", (provider_id, request.user_uuid))
            conn.execute(
                "DELETE FROM transactions WHERE truelayer_account_id = ? AND user_uuid = ?", (provider_id, request.user_uuid))
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
