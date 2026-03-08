import os
import requests
import uuid
import sqlite3
from urllib.parse import urlencode
from dotenv import load_dotenv
from cryptography.fernet import Fernet


class AccessTokenGenerator:
    def __init__(self, db_path="budai_memory.db"):
        load_dotenv()
        self.db_path = db_path
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.redirect_uri = os.getenv("REDIRECT_URI")
        self.auth_base_url = os.getenv("AUTH_LINK_URL")
        self.token_url = "https://auth.truelayer.com/connect/token"

        enc_key = os.getenv(
            "ENCRYPTION_KEY", b'cw_8H_1M4bX_3nF8vO5n3Y7A8xQ3_1m8aT2vP5_v5r8=')
        self.cipher_suite = Fernet(enc_key)

    def get_auth_link(self, user_uuid):
        state_uuid = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO auth_states (state_uuid, user_uuid) VALUES (?, ?)", (state_uuid, user_uuid))

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "scope": "info accounts balance cards transactions direct_debits standing_orders offline_access",
            "redirect_uri": self.redirect_uri,
            "providers": "uk-ob-all uk-oauth-all",
            "state": state_uuid
        }
        return f"{self.auth_base_url}?{urlencode(params)}"

    def generate_token_from_code(self, code, user_uuid):
        payload = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
            "code": code
        }
        response = requests.post(self.token_url, data=payload)
        res = response.json()
        if "access_token" in res:
            headers = {"Authorization": f"Bearer {res['access_token']}"}
            me_res = requests.get(
                "https://api.truelayer.com/data/v1/me", headers=headers).json()
            provider_id = me_res['results'][0]['provider']['provider_id']
            provider_name = me_res['results'][0]['provider']['display_name']

            enc_access = self.cipher_suite.encrypt(
                res["access_token"].encode())
            enc_refresh = self.cipher_suite.encrypt(
                res["refresh_token"].encode())

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO accounts (truelayer_account_id, user_uuid, bank_name, access_token, refresh_token, access_token_validity_time, access_status)
                    VALUES (?, ?, ?, ?, ?, datetime('now', '+90 days'), 'active')
                """, (provider_id, user_uuid, provider_name, enc_access, enc_refresh))
            return True
        return False

    def refresh_token(self, provider_id, user_uuid):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT refresh_token FROM accounts WHERE truelayer_account_id = ? AND user_uuid = ?", (provider_id, user_uuid))
            row = cursor.fetchone()
            if not row:
                return None

            refresh_token = self.cipher_suite.decrypt(row[0]).decode()

        payload = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token
        }
        response = requests.post(self.token_url, data=payload)
        res = response.json()
        if "access_token" in res:
            enc_access = self.cipher_suite.encrypt(
                res["access_token"].encode())
            enc_refresh = self.cipher_suite.encrypt(
                res.get("refresh_token", refresh_token).encode())

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE accounts SET access_token = ?, refresh_token = ? WHERE truelayer_account_id = ? AND user_uuid = ?
                """, (enc_access, enc_refresh, provider_id, user_uuid))
            return res["access_token"]
        return None

    def extend_connection(self, refresh_token, user_has_reconfirmed_consent=True):
        payload = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "user_has_reconfirmed_consent": user_has_reconfirmed_consent,
            "refresh_token": refresh_token
        }
        extend_url = "https://api.truelayer.com/data/v1/connections/extend"
        response = requests.post(extend_url, json=payload)
        return response.json()
