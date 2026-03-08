from flask import Blueprint, request, jsonify, redirect
from services.api_integrator.access_token_generator import AccessTokenGenerator
from middleware.auth_middleware import token_required
from cryptography.fernet import Fernet
import sqlite3
import os
import uuid

# Blueprint for standard auth routes (will be prefixed with /api/auth)
auth_bp = Blueprint('auth', __name__)

# NEW: Dedicated blueprint for the callback (will have NO prefix)
callback_bp = Blueprint('callback', __name__)

enc_key = os.getenv(
    "ENCRYPTION_KEY", b'cw_8H_1M4bX_3nF8vO5n3Y7A8xQ3_1m8aT2vP5_v5r8=')
cipher_suite = Fernet(enc_key)


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    with sqlite3.connect("budai_memory.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_uuid FROM users WHERE name = ? AND password = ?", (email, password))
        user = cursor.fetchone()

    if user:
        return jsonify({"token": user[0], "status": "success"})

    new_uuid = str(uuid.uuid4())
    with sqlite3.connect("budai_memory.db") as conn:
        conn.execute("INSERT INTO users (user_uuid, name, password) VALUES (?, ?, ?)",
                     (new_uuid, email, password))
    return jsonify({"token": new_uuid, "status": "success"})


@auth_bp.route('/truelayer/status', methods=['GET'])
@token_required
def truelayer_status():
    token_gen = AccessTokenGenerator()
    return jsonify({"auth_url": token_gen.get_auth_link(request.user_uuid)})


# MOVED to callback_bp: This will now listen exactly at http://localhost:8080/callback
@callback_bp.route('/callback', methods=['GET'])
def truelayer_callback():
    code = request.args.get('code')
    state = request.args.get('state')

    with sqlite3.connect("budai_memory.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_uuid FROM auth_states WHERE state_uuid = ?", (state,))
        row = cursor.fetchone()

    if not row:
        return jsonify({"error": "CSRF validation failed or session expired"}), 403

    user_uuid = row[0]
    token_gen = AccessTokenGenerator()
    if token_gen.generate_token_from_code(code, user_uuid):
        with sqlite3.connect("budai_memory.db") as conn:
            conn.execute(
                "DELETE FROM auth_states WHERE state_uuid = ?", (state,))
        # Redirect back to the Next.js Dashboard on port 3000
        return redirect("http://localhost:3000/dashboard")

    return jsonify({"error": "Authentication failed"}), 400


@auth_bp.route('/connections/extend', methods=['POST'])
@token_required
def extend_user_connections():
    data = request.json
    provider_ids = data.get('provider_ids', [])
    results = []

    token_gen = AccessTokenGenerator()

    with sqlite3.connect("budai_memory.db") as conn:
        cursor = conn.cursor()
        for p_id in provider_ids:
            cursor.execute(
                "SELECT refresh_token FROM accounts WHERE truelayer_account_id = ? AND user_uuid = ?", (p_id, request.user_uuid))
            row = cursor.fetchone()
            if not row:
                continue

            raw_refresh = cipher_suite.decrypt(row[0]).decode()
            res = token_gen.extend_connection(raw_refresh, True)

            if res.get("action_needed") == "no_action_needed":
                enc_access = cipher_suite.encrypt(res["access_token"].encode())
                enc_refresh = cipher_suite.encrypt(
                    res["refresh_token"].encode())
                conn.execute(
                    "UPDATE accounts SET access_token=?, refresh_token=?, access_token_validity_time=datetime('now', '+90 days') WHERE truelayer_account_id=?", (enc_access, enc_refresh, p_id))
                results.append({"provider_id": p_id, "status": "success"})
            elif res.get("action_needed") == "authentication_needed":
                results.append({"provider_id": p_id, "status": "requires_reauth",
                               "redirect_url": res.get("user_input_link")})

    return jsonify({"results": results})
