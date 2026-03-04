import requests
import os
import pandas as pd
import sqlite3


class UserAccount:
    def __init__(self, identifier=None, db_path="budai_memory.db"):
        self.base_url = "https://api.truelayer.com/data/v1/accounts"
        self.headers = {"accept": "application/json"}
        self.account_id = None
        self.access_token = None
        self.db_path = db_path

        if identifier:
            self.resolve_account(identifier)
        else:
            self.set_default_account()

    def resolve_account(self, identifier):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT provider_name, access_token FROM credentials")
                rows = cursor.fetchall()

            if not rows:
                return

            id_lower = str(identifier).lower()

            # Check for Bank Name Match
            for provider_name, token in rows:
                if id_lower in provider_name.lower():
                    headers = {"Authorization": f"Bearer {token}",
                               "accept": "application/json"}
                    res = requests.get(self.base_url, headers=headers)
                    if res.status_code == 200 and res.json().get("results"):
                        self.account_id = res.json(
                        )["results"][0]["account_id"]
                        self.access_token = token
                        self.headers["Authorization"] = f"Bearer {token}"
                        return

            # Check for Explicit Account ID Match
            for _, token in rows:
                headers = {"Authorization": f"Bearer {token}",
                           "accept": "application/json"}
                res = requests.get(
                    f"{self.base_url}/{identifier}/balance", headers=headers)
                if res.status_code == 200:
                    self.account_id = identifier
                    self.access_token = token
                    self.headers["Authorization"] = f"Bearer {token}"
                    return

            self.set_default_account()
        except Exception:
            self.set_default_account()

    def set_default_account(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT access_token FROM credentials LIMIT 1")
                row = cursor.fetchone()
            if row:
                self.access_token = row[0]
                self.headers["Authorization"] = f"Bearer {self.access_token}"
                res = requests.get(self.base_url, headers=self.headers)
                if res.status_code == 200 and res.json().get("results"):
                    self.account_id = res.json()["results"][0]["account_id"]
        except Exception:
            pass

    def all_transactions(self, from_date=None, to_date=None):
        if not self.account_id:
            return None
        url = f"{self.base_url}/{self.account_id}/transactions"
        params = {"from": from_date, "to": to_date}
        try:
            res = requests.get(url, headers=self.headers, params=params)
            if res.status_code == 200:
                results = res.json().get("results", [])
                return pd.DataFrame(results) if results else None
        except Exception:
            return None

    def get_account_balance(self):
        if not self.account_id:
            return None
        url = f"{self.base_url}/{self.account_id}/balance"
        try:
            res = requests.get(url, headers=self.headers)
            if res.status_code == 200:
                return res.json().get("results", {})
        except Exception:
            return None
