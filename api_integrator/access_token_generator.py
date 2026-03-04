import os
import requests
import uuid
import sqlite3
from urllib.parse import urlencode
from dotenv import load_dotenv


class AccessTokenGenerator:
    def __init__(self, db_path="budai_memory.db"):
        load_dotenv()
        self.db_path = db_path
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.redirect_uri = os.getenv("REDIRECT_URI")
        self.auth_base_url = os.getenv("AUTH_LINK_URL")
        self.token_url = "https://auth.truelayer.com/connect/token"

    def get_auth_link(self):
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "scope": "info accounts balance cards transactions direct_debits standing_orders offline_access",
            "redirect_uri": self.redirect_uri,
            "providers": "uk-ob-all uk-oauth-all",
            "state": str(uuid.uuid4())
        }
        return f"{self.auth_base_url}?{urlencode(params)}"

    def generate_token_from_code(self, code):
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

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO credentials (provider_id, provider_name, access_token, refresh_token)
                    VALUES (?, ?, ?, ?)
                """, (provider_id, provider_name, res["access_token"], res["refresh_token"]))
            return True
        return False

    def refresh_token(self, provider_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT refresh_token FROM credentials WHERE provider_id = ?", (provider_id,))
            row = cursor.fetchone()
            if not row:
                return None
            refresh_token = row[0]

        payload = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token
        }
        response = requests.post(self.token_url, data=payload)
        res = response.json()
        if "access_token" in res:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE credentials SET access_token = ?, refresh_token = ? WHERE provider_id = ?
                """, (res["access_token"], res.get("refresh_token", refresh_token), provider_id))
            return res["access_token"]
        return None
