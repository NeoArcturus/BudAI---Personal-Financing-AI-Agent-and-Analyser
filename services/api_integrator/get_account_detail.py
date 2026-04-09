import requests
import traceback
import time
import uuid
import pandas as pd
from datetime import datetime
from cryptography.fernet import Fernet
from services.api_integrator.access_token_generator import AccessTokenGenerator
from config import SessionLocal, TRUELAYER_BASE_URL, ENCRYPTION_KEY
from models.database_models import Account, Bank, Transaction
from sqlalchemy import text


class UserAccounts:
    def __init__(self, user_id=None):
        self.base_url = f"{TRUELAYER_BASE_URL}/accounts"
        self.user_id = user_id
        self.cipher_suite = Fernet(ENCRYPTION_KEY)

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
            token_gen = AccessTokenGenerator()
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

    def initialise_accounts(self, bank_uuid, user_uuid):
        try:
            with SessionLocal() as session:
                bank = session.query(Bank).filter_by(
                    bank_uuid=bank_uuid, user_uuid=user_uuid).first()

                if not bank:
                    print(
                        f"[BACKEND LOG] Bank UUID {bank_uuid} not found during initialization.")
                    return False

                access_token_raw = bank.access_token
                if isinstance(access_token_raw, memoryview):
                    access_token_raw = access_token_raw.tobytes()

                access_token = self.cipher_suite.decrypt(
                    access_token_raw).decode()
                provider_id = bank.truelayer_provider_id

                account_res = self._make_request(
                    self.base_url, access_token, provider_id)

                if account_res is not None and account_res.status_code == 200:
                    results = account_res.json().get("results", [])
                    if not results:
                        return False

                    for acc_det in results:
                        acc_id = acc_det.get("account_id")

                        account_number_info = acc_det.get("account_number")
                        if isinstance(account_number_info, list) and len(account_number_info) > 0:
                            account_number_info = account_number_info[0]
                        if not isinstance(account_number_info, dict):
                            account_number_info = {}

                        sort_code = str(
                            account_number_info.get("sort_code", ""))
                        acc_no = str(account_number_info.get("number", ""))

                        acc_balance = 0.0
                        bal_res = self._make_request(
                            f"{self.base_url}/{acc_id}/balance", access_token, provider_id)
                        if bal_res is not None and bal_res.status_code == 200:
                            bal_data = bal_res.json().get("results", [{}])[0]
                            acc_balance = bal_data.get(
                                "available", bal_data.get("current", 0.0))

                        # ORM INSERTION
                        existing_acc = session.query(Account).filter_by(
                            account_id=acc_id).first()

                        if existing_acc:
                            existing_acc.user_uuid = user_uuid
                            existing_acc.bank_uuid = bank_uuid
                            existing_acc.account_number = acc_no
                            existing_acc.sort_code = sort_code
                            existing_acc.account_balance = float(acc_balance)
                        else:
                            new_acc = Account(
                                account_id=acc_id,
                                user_uuid=user_uuid,
                                bank_uuid=bank_uuid,
                                account_number=acc_no,
                                sort_code=sort_code,
                                account_balance=float(acc_balance)
                            )
                            session.add(new_acc)

                        # Commit the account first so transaction foreign keys don't fail
                        session.commit()

                        # --- SYNC 2025 TRANSACTIONS IMMEDIATELY ---
                        print(
                            f"[SYNC] Fetching 2025 transactions for {acc_id}...")
                        tx_url = f"{self.base_url}/{acc_id}/transactions"
                        tx_params = {"from": "2025-01-01", "to": "2025-12-31"}
                        tx_res = self._make_request(
                            tx_url, access_token, provider_id, params=tx_params)

                        if tx_res is not None and tx_res.status_code == 200:
                            tx_data = tx_res.json().get("results", [])
                            for tx in tx_data:
                                tx_id = tx.get("transaction_id",
                                               str(uuid.uuid4()))
                                date_str = tx.get("timestamp")

                                if date_str:
                                    try:
                                        date_val = datetime.fromisoformat(
                                            date_str.replace("Z", "+00:00"))
                                    except Exception:
                                        date_val = datetime.utcnow()
                                else:
                                    date_val = datetime.utcnow()

                                amount = float(tx.get("amount", 0.0))
                                original_desc = str(tx.get("description", ""))
                                classification_list = tx.get(
                                    "transaction_classification", [])

                                if isinstance(classification_list, list) and classification_list:
                                    classification_str = " ".join(
                                        [str(c) for c in classification_list])
                                    desc_val = f"{original_desc} {classification_str}".strip(
                                    )
                                else:
                                    desc_val = original_desc

                                existing_tx = session.query(Transaction).filter_by(
                                    transaction_uuid=tx_id).first()
                                if not existing_tx:
                                    new_tx = Transaction(
                                        transaction_uuid=tx_id,
                                        user_uuid=user_uuid,
                                        bank_uuid=bank_uuid,
                                        account_id=acc_id,
                                        date=date_val,
                                        amount=amount,
                                        category="Uncategorized",
                                        description=desc_val
                                    )
                                    session.add(new_tx)
                            session.commit()

                    return True
                return False
        except Exception as e:
            print("[CRITICAL ERROR] Initialising Accounts:")
            traceback.print_exc()
            return False

    def get_all_accounts(self):
        all_accounts = []
        try:
            with SessionLocal() as session:
                banks = session.query(Bank).filter_by(
                    user_uuid=self.user_id).all()

            for bank in banks:
                self.initialise_accounts(bank.bank_uuid, self.user_id)

            with SessionLocal() as session:
                updated_banks = session.query(Bank).filter_by(
                    user_uuid=self.user_id).all()
                for b in updated_banks:
                    if b.consent_status == 'revoked':
                        all_accounts.append({
                            "account_id": b.truelayer_provider_id,
                            "provider_name": b.bank_name,
                            "account_number": "****",
                            "sort_code": "00-00-00",
                            "currency": "GBP",
                            "balance": 0.0,
                            "status": "revoked",
                            "provider_id": b.truelayer_provider_id
                        })
                        continue

                    for acc in b.accounts:
                        all_accounts.append({
                            "account_id": acc.account_id,
                            "provider_name": b.bank_name,
                            "account_number": acc.account_number[-4:] if acc.account_number else "****",
                            "sort_code": acc.sort_code or "",
                            "currency": "GBP",
                            "balance": acc.account_balance or 0.0,
                            "status": "active",
                            "provider_id": b.truelayer_provider_id
                        })

            return all_accounts
        except Exception:
            traceback.print_exc()
            return []

    def get_account_balance(self, bank_name_or_id, user_uuid, account_type="TRANSACTION"):
        with SessionLocal() as session:
            acc = session.query(Account).join(Bank).filter(
                (Bank.bank_name == bank_name_or_id) | (
                    Account.account_id == bank_name_or_id),
                Account.user_uuid == user_uuid
            ).first()

        if acc and acc.account_balance is not None:
            return float(acc.account_balance)
        return 0.0

    def get_transactions(self, bank_name_or_id, user_uuid, start_date=None, end_date=None):
        try:
            with SessionLocal() as session:
                query = session.query(Transaction).join(Account).join(Bank).filter(
                    (Bank.bank_name == bank_name_or_id) | (
                        Account.account_id == bank_name_or_id),
                    Transaction.user_uuid == user_uuid
                )

                if start_date:
                    query = query.filter(Transaction.date >= start_date)
                if end_date:
                    query = query.filter(Transaction.date <= end_date)

                txs = query.all()

                if not txs:
                    return pd.DataFrame()

                transactions = []
                for tx in txs:
                    transactions.append({
                        "transaction_id": tx.transaction_uuid,
                        "timestamp": tx.date.isoformat() if tx.date else None,
                        "date": tx.date.isoformat() if tx.date else None,
                        "amount": tx.amount,
                        "description": tx.description,
                        "Category": tx.category,
                        "bank_uuid": tx.bank_uuid,
                        "account_id": tx.account_id
                    })

                return pd.DataFrame(transactions)
        except Exception:
            traceback.print_exc()
            return pd.DataFrame()

    def get_bank_transactions(self, bank_name_or_id: str, user_uuid: str, from_date: str, to_date: str):
        try:
            print(
                "[TRUELAYER API BACKEND LOG] Entering get_bank_transactions function...")
            print(
                f"Params: bank_name_or_id: {bank_name_or_id}, user_uuid: {user_uuid}, from_date: {from_date}, to_date: {to_date}")
            transactions = self.get_transactions(
                bank_name_or_id, user_uuid, start_date=from_date, end_date=to_date)

            needs_api_fetch = True

            if transactions is not None and not transactions.empty:
                temp_dates = pd.to_datetime(
                    transactions['date'], errors='coerce', utc=True)
                oldest_db_record = temp_dates.min()
                requested_start = pd.to_datetime(from_date, utc=True)

                if oldest_db_record <= requested_start + pd.Timedelta(days=5):
                    needs_api_fetch = False

            if not needs_api_fetch:
                return transactions

            with SessionLocal() as session:
                query = text("""
                                SELECT a.account_id, b.access_token, b.truelayer_provider_id
                                FROM banks b
                                JOIN accounts a ON b.bank_uuid = a.bank_uuid
                                WHERE (b.bank_name = :identifier OR a.account_id = :identifier) 
                                AND a.user_uuid = :user_uuid
                            """)
                rows = session.execute(
                    query, {"identifier": bank_name_or_id, "user_uuid": user_uuid}).fetchall()

                if not rows:
                    print("[TRUELAYER API BACKEND LOG] Account does not exist!")
                    return pd.DataFrame()

            all_txs = []

            for row in rows:
                print(f"[TRUELAYER API BACKEND LOG] Row data: {row}")
                account_id = row[0]
                access_token = self.cipher_suite.decrypt(row[1]).decode()
                provider_id = row[2]

                url = self.base_url + f"/{account_id}/transactions"
                params = {"from": from_date, "to": to_date}

                result = self._make_request(
                    url=url, token=access_token, provider_id=provider_id, params=params)

                print(
                    f"[TRUELAYER API BACKEND LOG] API request result status code: {result.status_code}")
                if result and result.status_code == 200:
                    data = result.json()
                    transactions_list = data.get(
                        'results', data) if isinstance(data, dict) else data
                    all_txs.extend(transactions_list)

            return pd.DataFrame(all_txs)

        except Exception as e:
            print(f"Error fetching bank transactions: {e}")
            return pd.DataFrame()

    def get_transactions_by_account(self, account_id):
        try:
            with SessionLocal() as session:
                txs = session.query(Transaction).filter_by(
                    account_id=account_id, user_uuid=self.user_id).order_by(Transaction.date.desc()).all()
                if not txs:
                    return []

                results = []
                for tx in txs:
                    results.append({
                        "transaction_id": tx.transaction_uuid,
                        "timestamp": tx.date.isoformat() if tx.date else None,
                        "amount": tx.amount,
                        "description": tx.description,
                        "category": tx.category
                    })
                return results
        except Exception:
            traceback.print_exc()
            return []

    def revoke_provider_connection(self, provider_id):
        try:
            with SessionLocal() as session:
                bank = session.query(Bank).filter_by(
                    truelayer_provider_id=provider_id, user_uuid=self.user_id).first()
                if bank:
                    for acc in bank.accounts:
                        session.delete(acc)
                    session.delete(bank)
                    session.commit()
            return True
        except Exception:
            traceback.print_exc()
            return False
