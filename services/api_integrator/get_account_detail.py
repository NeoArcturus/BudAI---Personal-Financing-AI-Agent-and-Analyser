import requests
import os
import pandas as pd
import sqlite3
import time
from cryptography.fernet import Fernet
from services.api_integrator.access_token_generator import AccessTokenGenerator
from dotenv import load_dotenv


class UserAccounts:
    def __init__(self, user_id=None, db_path="budai_memory.db"):
        load_dotenv()
        self.base_url = "https://api.truelayer.com/data/v1/accounts"
        self.user_id = user_id
        self.db_path = db_path
        enc_key = os.getenv(
            "ENCRYPTION_KEY", b'cw_8H_1M4bX_3nF8vO5n3Y7A8xQ3_1m8aT2vP5_v5r8=')
        self.cipher_suite = Fernet(enc_key)

    def _make_request(self, url, token, provider_id, params=None, max_retries=3):
        headers = {"accept": "application/json",
                   "Authorization": f"Bearer {token}"}
        res = None
        for attempt in range(max_retries):
            res = requests.get(url, headers=headers, params=params)
            if res.status_code != 429:
                break
            time.sleep(2 ** attempt)

        if res is not None and res.status_code == 401 and provider_id:
            token_gen = AccessTokenGenerator(self.db_path)
            new_token = token_gen.refresh_token(provider_id, self.user_id)
            if new_token:
                headers["Authorization"] = f"Bearer {new_token}"
                res = requests.get(url, headers=headers, params=params)

        if res is not None and res.status_code == 403:
            err_data = res.json()
            if isinstance(err_data, dict) and (err_data.get("error") == "sca_exceeded" or "PSU" in str(err_data)):
                raise PermissionError(
                    "SECURITY LOCK: The bank blocked access to historical data (older than 90 days).")
        return res

    def initialise_accounts(self, bank_name, user_uuid):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT bank_uuid, access_token, truelayer_provider_id FROM banks WHERE bank_name = ? AND user_uuid = ?", (bank_name, user_uuid))
                row = cursor.fetchone()
                if not row:
                    return False
                bank_uuid = row[0]
                access_token = self.cipher_suite.decrypt(row[1]).decode()
                provider_id = row[2]

                account_res = self._make_request(
                    self.base_url, access_token, provider_id)

                if account_res is not None and account_res.status_code == 200:
                    results = account_res.json().get("results", [])
                    if not results:
                        return False
                    for acc_det in results:
                        acc_id = acc_det.get("account_id")
                        account_number_info = acc_det.get("account_number", {})
                        sort_code = account_number_info.get("sort_code")
                        acc_no = account_number_info.get("number")
                        acc_type = acc_det.get("account_type", "TRANSACTION")

                        acc_balance = 0.0
                        bal_res = self._make_request(
                            f"{self.base_url}/{acc_id}/balance", access_token, provider_id)
                        if bal_res is not None and bal_res.status_code == 200:
                            bal_data = bal_res.json().get("results", [{}])[0]
                            acc_balance = bal_data.get(
                                "available", bal_data.get("current", 0.0))

                        conn.execute("""
                            INSERT OR REPLACE INTO accounts (account_id, user_uuid, bank_uuid, account_number, sort_code, account_type, account_balance) 
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (acc_id, user_uuid, bank_uuid, acc_no, sort_code, acc_type, acc_balance))
                    conn.commit()
                    return True
                return False
        except Exception:
            return False

    def get_all_accounts(self):
        all_accounts = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT bank_name FROM banks WHERE user_uuid = ?", (self.user_id,))
                banks = cursor.fetchall()

            for bank in banks:
                self.initialise_accounts(bank[0], self.user_id)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT b.truelayer_provider_id, b.access_token, b.consent_status, b.bank_name, 
                           a.account_id, a.account_number, a.sort_code, a.account_balance
                    FROM banks b
                    LEFT JOIN accounts a ON b.bank_uuid = a.bank_uuid
                    WHERE b.user_uuid = ?
                """, (self.user_id,))
                rows = cursor.fetchall()

            for provider_id, enc_token, status, bank_name, acc_id, acc_num, sort_code, db_balance in rows:
                if status == 'revoked':
                    all_accounts.append({
                        "account_id": acc_id or provider_id,
                        "provider_name": bank_name,
                        "account_number": "****",
                        "sort_code": "00-00-00",
                        "currency": "GBP",
                        "balance": 0.0,
                        "status": "revoked",
                        "provider_id": provider_id
                    })
                    continue

                balance_val = db_balance if db_balance is not None else 0.0

                all_accounts.append({
                    "account_id": acc_id,
                    "provider_name": bank_name,
                    "account_number": acc_num[-4:] if acc_num else "****",
                    "sort_code": sort_code or "",
                    "currency": "GBP",
                    "balance": balance_val,
                    "status": "active",
                    "provider_id": provider_id
                })

            return all_accounts
        except Exception:
            return []

    def get_account_balance(self, bank_name_or_id, user_uuid, account_type="TRANSACTION"):
        """Fetches the balance directly from the local DB to prevent TrueLayer API rate-limiting."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT a.account_balance 
                FROM accounts a 
                JOIN banks b ON a.bank_uuid = b.bank_uuid 
                WHERE (b.bank_name = ? OR a.account_id = ?) AND a.user_uuid = ? AND a.account_type = ?
            """, (bank_name_or_id, bank_name_or_id, user_uuid, account_type))
            row = cursor.fetchone()

        if row and row[0] is not None:
            return float(row[0])

        return 0.0

    def get_transactions(self, bank_name_or_id, user_uuid, start_date=None, end_date=None, account_type="TRANSACTION"):
        print(
            f"[BACKEND LOG] get_transactions | input_identifier: {bank_name_or_id} | user_uuid: {user_uuid} | start_date: {start_date} | end_date: {end_date}")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT a.account_id, b.access_token, b.truelayer_provider_id 
                FROM accounts a 
                JOIN banks b ON a.bank_uuid = b.bank_uuid 
                WHERE (b.bank_name = ? OR a.account_id = ?) AND a.user_uuid = ? AND a.account_type = ?
            """, (bank_name_or_id, bank_name_or_id, user_uuid, account_type))
            row = cursor.fetchone()

            if not row:
                return pd.DataFrame()

            acc_id = row[0]
            access_token = self.cipher_suite.decrypt(row[1]).decode()
            provider_id = row[2]

            transaction_url = self.base_url + f"/{acc_id}/transactions"

            params = {}
            if start_date:
                params["from"] = start_date
            if end_date:
                params["to"] = end_date

            res = self._make_request(
                transaction_url, access_token, provider_id, params=params)

            if res is not None and res.status_code == 200:
                transactions_data = res.json().get("results", [])
                print(
                    f"[BACKEND LOG] TrueLayer returned {len(transactions_data)} transactions for Bank: {bank_name_or_id} and user: {user_uuid}")
                transactions = []

                for transaction in transactions_data:
                    date = transaction.get("timestamp")
                    amt = transaction.get("amount")

                    original_desc = str(transaction.get("description", ""))
                    classification_list = transaction.get(
                        "transaction_classification", [])

                    if isinstance(classification_list, list) and classification_list:
                        classification_str = " ".join(
                            [str(c) for c in classification_list])
                        transaction["description"] = f"{original_desc} {classification_str}".strip(
                        )
                        transaction["truelayer_classification"] = classification_str

                    transaction["bank_name"] = bank_name_or_id
                    transaction["account_id"] = acc_id
                    transactions.append(transaction)

                return pd.DataFrame(transactions)

            status = res.status_code if res is not None else "None"
            err_text = res.text if res is not None else "No Response Object"
            print(
                f"[BACKEND LOG] get_transactions | TrueLayer API failed with status: {status} | Error: {err_text}")

            return pd.DataFrame()

    def get_transactions_by_account(self, account_id):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT b.access_token, b.truelayer_provider_id FROM banks b
                    JOIN accounts a ON b.bank_uuid = a.bank_uuid
                    WHERE a.account_id = ? AND b.user_uuid = ?
                """, (account_id, self.user_id))
                row = cursor.fetchone()

            if row:
                access_token = self.cipher_suite.decrypt(row[0]).decode()
                provider_id = row[1]
                res = self._make_request(
                    f"{self.base_url}/{account_id}/transactions", access_token, provider_id)

                if res is not None and res.status_code == 200:
                    transactions = res.json().get("results", [])
                    df = pd.DataFrame(transactions)
                    if not df.empty:
                        return df.fillna("").to_dict(orient="records")
            return []
        except Exception:
            return []

    def revoke_provider_connection(self, provider_id):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT bank_uuid FROM banks WHERE truelayer_provider_id = ? AND user_uuid = ?", (provider_id, self.user_id))
                row = cursor.fetchone()
                if row:
                    bank_uuid = row[0]
                    conn.execute(
                        "DELETE FROM transactions WHERE bank_uuid = ?", (bank_uuid,))
                    conn.execute(
                        "DELETE FROM accounts WHERE bank_uuid = ?", (bank_uuid,))
                    conn.execute(
                        "DELETE FROM banks WHERE bank_uuid = ?", (bank_uuid,))
            return True
        except Exception:
            return False
