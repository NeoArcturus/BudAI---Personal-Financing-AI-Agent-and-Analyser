import requests
import os
import pandas as pd
import sqlite3
import time
from cryptography.fernet import Fernet
from services.api_integrator.access_token_generator import AccessTokenGenerator
from dotenv import load_dotenv


class UserAccount:
    def __init__(self, identifier=None, user_id=None, db_path="budai_memory.db"):
        load_dotenv()
        self.base_url = "https://api.truelayer.com/data/v1/accounts"
        self.headers = {"accept": "application/json"}
        self.account_id = None
        self.access_token = None
        self.provider_id = None
        self.user_id = user_id
        self.db_path = db_path
        self.needs_user_clarification = False

        enc_key = os.getenv(
            "ENCRYPTION_KEY", b'cw_8H_1M4bX_3nF8vO5n3Y7A8xQ3_1m8aT2vP5_v5r8=')
        self.cipher_suite = Fernet(enc_key)

        self._initialize_account(identifier)

    def _initialize_account(self, identifier):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT truelayer_account_id, bank_name, access_token, user_uuid FROM accounts"
                )
                rows = cursor.fetchall()

            if rows:
                self.user_id = rows[0][3]

            print(
                f"\n[BACKEND LOG] DB Account Check -> User ID: {self.user_id} | Linked Accounts Found: {len(rows)}")

            if not rows:
                print("[BACKEND LOG] No accounts found in the database.")
                return

            if identifier and str(identifier).lower() != "none":
                self.resolve_account(identifier, rows)
            else:
                if len(rows) == 1:
                    print(
                        f"[BACKEND LOG] Only 1 account found. Auto-selecting: {rows[0][1]}")
                    self._set_account_data(rows[0])
                elif len(rows) > 1:
                    print(
                        f"[BACKEND LOG] Multiple accounts found. Triggering user clarification flag.")
                    self.needs_user_clarification = True

        except Exception as e:
            print(f"[BACKEND ERROR] {e}")

    def resolve_account(self, identifier, rows):
        id_lower = str(identifier).lower()
        for p_id, p_name, enc_token, u_id in rows:
            if id_lower in p_name.lower() or id_lower in p_id.lower():
                self._set_account_data((p_id, p_name, enc_token, u_id))
                return

        if len(rows) == 1:
            self._set_account_data(rows[0])
        else:
            self.needs_user_clarification = True

    def _set_account_data(self, row_data):
        p_id, p_name, enc_token, u_id = row_data
        token = self.cipher_suite.decrypt(enc_token).decode()
        self.provider_id = p_id
        self.access_token = token
        self.headers["Authorization"] = f"Bearer {token}"

        res = self._make_request(self.base_url)
        if res and res.status_code == 200 and res.json().get("results"):
            self.account_id = res.json()["results"][0]["account_id"]

    def _make_request(self, url, params=None, max_retries=3):
        res = None
        for attempt in range(max_retries):
            res = requests.get(url, headers=self.headers, params=params)
            if res.status_code != 429:
                break
            time.sleep(2 ** attempt)

        if res and res.status_code == 401 and self.provider_id:
            token_gen = AccessTokenGenerator(self.db_path)
            new_token = token_gen.refresh_token(self.provider_id, self.user_id)
            if new_token:
                self.access_token = new_token
                self.headers["Authorization"] = f"Bearer {new_token}"
                res = requests.get(url, headers=self.headers, params=params)

        if res and res.status_code == 403:
            err_data = res.json()
            if err_data.get("error") == "sca_exceeded" or "PSU" in str(err_data):
                raise PermissionError(
                    "SECURITY LOCK: The bank blocked access to historical data (older than 90 days)."
                )
        return res

    def all_transactions(self, from_date=None, to_date=None):
        if not self.account_id:
            return None
        url = f"{self.base_url}/{self.account_id}/transactions"
        params = {}
        if from_date and to_date:
            params = {"from": from_date, "to": to_date}
        res = self._make_request(url, params=params)
        if res and res.status_code == 200:
            results = res.json().get("results", [])
            return pd.DataFrame(results) if results else None
        return None

    def get_account_balance(self):
        if not self.account_id:
            return None
        url = f"{self.base_url}/{self.account_id}/balance"
        res = self._make_request(url)
        if res and res.status_code == 200:
            return res.json().get("results", {})
        return None
