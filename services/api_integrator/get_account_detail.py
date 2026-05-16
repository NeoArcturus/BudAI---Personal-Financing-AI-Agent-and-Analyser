import requests
import time
import uuid
import pandas as pd
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from services.api_integrator.access_token_generator import AccessTokenGenerator
from config import SessionLocal, TRUELAYER_BASE_URL, ENCRYPTION_KEY
from models.database_models import Account, Bank, Transaction
from sqlalchemy import text
import logging

logger = logging.getLogger("uvicorn.error")


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
            logger.info(f"Making request to {url} (Attempt {attempt + 1})")
            res = requests.get(url, headers=headers, params=params)
            if res.status_code != 429:
                break
            logger.warning(
                f"Rate limited (429) for {url}. Retrying in {2**attempt}s...")
            time.sleep(2 ** attempt)

        if res is not None and res.status_code == 401 and provider_id:
            logger.info(
                f"Token expired for {provider_id}. Attempting refresh...")
            token_gen = AccessTokenGenerator()
            new_token = token_gen.refresh_token(provider_id, self.user_id)
            if new_token:
                logger.info(f"Token refreshed successfully for {provider_id}")
                headers["Authorization"] = f"Bearer {new_token}"
                res = requests.get(url, headers=headers, params=params)

        if res is not None and res.status_code == 403:
            err_data = res.json()
            logger.error(f"Access forbidden (403) for {url}: {err_data}")
            if isinstance(err_data, dict) and (err_data.get("error") == "sca_exceeded" or "PSU" in str(err_data)):
                raise PermissionError("SECURITY LOCK")

        if res is not None:
            logger.info(
                f"Request to {url} completed with status {res.status_code}")

        return res

    def initialise_accounts(self, bank_uuid, user_uuid):
        logger.info(
            f"Starting account initialization for Bank: {bank_uuid}, User: {user_uuid}")
        try:
            with SessionLocal() as session:
                bank = session.query(Bank).filter_by(
                    bank_uuid=bank_uuid, user_uuid=user_uuid).first()

                if not bank:
                    logger.warning(f"Bank connection not found: {bank_uuid}")
                    return False

                logger.info(
                    f"Decrypting tokens for bank: {bank.bank_name or bank.truelayer_provider_id}")
                access_token_raw = bank.access_token
                if isinstance(access_token_raw, memoryview):
                    access_token_raw = access_token_raw.tobytes()

                access_token = self.cipher_suite.decrypt(
                    access_token_raw).decode()
                provider_id = bank.truelayer_provider_id

                logger.info(
                    f"Fetching accounts from TrueLayer for provider: {provider_id}")
                account_res = self._make_request(
                    self.base_url, access_token, provider_id)

                if account_res is not None and account_res.status_code == 200:
                    results = account_res.json().get("results", [])
                    logger.info(
                        f"Found {len(results)} accounts for provider {provider_id}")
                    if not results:
                        return False

                    for acc_det in results:
                        acc_id = acc_det.get("account_id")
                        display_name = acc_det.get(
                            "display_name", "Unknown Account")
                        logger.info(
                            f"Processing account: {display_name} ({acc_id})")

                        account_number_info = acc_det.get("account_number")
                        if isinstance(account_number_info, list) and len(account_number_info) > 0:
                            account_number_info = account_number_info[0]
                        if not isinstance(account_number_info, dict):
                            account_number_info = {}

                        sort_code = str(
                            account_number_info.get("sort_code", ""))
                        acc_no = str(account_number_info.get("number", ""))

                        logger.info(f"Fetching balance for account: {acc_id}")
                        acc_balance = 0.0
                        bal_res = self._make_request(
                            f"{self.base_url}/{acc_id}/balance", access_token, provider_id)
                        if bal_res is not None and bal_res.status_code == 200:
                            bal_data = bal_res.json().get("results", [{}])[0]
                            acc_balance = bal_data.get(
                                "available", bal_data.get("current", 0.0))
                            logger.info(
                                f"Account {acc_id} balance: {acc_balance}")

                        existing_acc = session.query(Account).filter_by(
                            account_id=acc_id).first()

                        if existing_acc:
                            logger.info(
                                f"Updating existing account record: {acc_id}")
                            existing_acc.user_uuid = user_uuid
                            existing_acc.bank_uuid = bank_uuid
                            existing_acc.account_number = acc_no
                            existing_acc.sort_code = sort_code
                            existing_acc.account_balance = float(acc_balance)
                        else:
                            logger.info(
                                f"Creating new account record: {acc_id}")
                            new_acc = Account(
                                account_id=acc_id,
                                user_uuid=user_uuid,
                                bank_uuid=bank_uuid,
                                account_number=acc_no,
                                sort_code=sort_code,
                                account_balance=float(acc_balance)
                            )
                            session.add(new_acc)

                        session.commit()

                        logger.info(
                            f"Fetching transactions for account: {acc_id}")
                        tx_url = f"{self.base_url}/{acc_id}/transactions"
                        from_date = (datetime.utcnow() -
                                     timedelta(days=180)).strftime('%Y-%m-%d')
                        to_date = datetime.utcnow().strftime('%Y-%m-%d')
                        tx_params = {"from": from_date, "to": to_date}
                        tx_res = self._make_request(
                            tx_url, access_token, provider_id, params=tx_params)

                        if tx_res is not None and tx_res.status_code == 200:
                            tx_data = tx_res.json().get("results", [])
                            logger.info(
                                f"Retrieved {len(tx_data)} raw transactions for account {acc_id}")
                            self._process_and_store_transactions(
                                session, tx_data, user_uuid, bank_uuid, acc_id)

                    logger.info(
                        f"Finished initialization for bank connection: {bank_uuid}")
                    return True

                logger.warning(
                    f"Failed to fetch accounts for provider {provider_id}. Status: {account_res.status_code if account_res else 'No response'}")
                return False
        except Exception as e:
            logger.error(f"Error initializing accounts: {e}", exc_info=True)
            return False

    def _process_and_store_transactions(self, session, tx_data, user_uuid, bank_uuid, account_id):
        import hashlib
        from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
        from services.Categorizer_Agent.categorizer.preprocessor import Preprocessor
        import os

        if not tx_data:
            logger.info(
                f"No transaction data to process for account {account_id}")
            return

        logger.info(
            f"Processing {len(tx_data)} transactions for account {account_id}")
        new_txs = []
        for tx in tx_data:
            tx_id = tx.get("transaction_id", str(uuid.uuid4()))
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
            classification_list = tx.get("transaction_classification", [])

            if isinstance(classification_list, list) and classification_list:
                classification_str = " ".join(
                    [str(c) for c in classification_list])
                desc_val = f"{original_desc} {classification_str}".strip()
            else:
                desc_val = original_desc

            # Content-based Deduplication Hash
            tx_hash = hashlib.sha256(
                f"{user_uuid}_{account_id}_{date_val.strftime('%Y-%m-%d')}_{amount}_{desc_val}".encode()).hexdigest()

            # Check if exists by UUID or Hash
            existing_tx = session.query(Transaction).filter(
                (Transaction.transaction_uuid == tx_id) |
                (Transaction.transaction_uuid == tx_hash)
            ).first()

            if not existing_tx:
                new_txs.append({
                    "transaction_uuid": tx_id,
                    "user_uuid": user_uuid,
                    "bank_uuid": bank_uuid,
                    "account_id": account_id,
                    "date": date_val,
                    "amount": amount,
                    "description": desc_val,
                    "category": "Uncategorized"
                })

        if not new_txs:
            logger.info(
                f"All {len(tx_data)} transactions for account {account_id} are already in the database. Skipping.")
            return

        logger.info(
            f"Found {len(new_txs)} new transactions to categorize and store for account {account_id}")

        # Prepare for Categorization
        df_new = pd.DataFrame(new_txs)
        agent = CategorizerAgent()
        proc = Preprocessor(df_new, agent.local_st_path)

        xgb_model_path = os.path.join(agent.model_dir, "gbm_model.joblib")
        enc_path = os.path.join(agent.enc_dir, "label_encoder.joblib")

        if os.path.exists(xgb_model_path) and os.path.exists(enc_path):
            logger.info(
                f"Running AI Categorization for {len(new_txs)} transactions...")
            clean_df, embeddings = proc.preprocess_for_inference()
            categorized_df = agent.categorizer.predict(
                clean_df, embeddings, xgb_model_path, enc_path)
            # Map categories back
            category_map = categorized_df.set_index(
                "transaction_uuid")["Category"].to_dict()
            for tx in new_txs:
                tx["category"] = category_map.get(
                    tx["transaction_uuid"], "Uncategorized")
            logger.info("AI Categorization completed.")
        else:
            logger.warning(
                f"Categorization models not found at {xgb_model_path}. Storing as 'Uncategorized'.")

        # Sort by date (chronological) before storing
        new_txs.sort(key=lambda x: x["date"])

        # Bulk store
        logger.info(f"Committing {len(new_txs)} transactions to database...")
        # Bulk store
        for tx_dict in new_txs:
            new_tx = Transaction(**tx_dict)
            session.add(new_tx)

        session.commit()

        # Phase 6: Index in ChromaDB
        from services.memory_service import MemoryService
        try:
            mem = MemoryService()
            mem.index_transactions(new_txs, user_uuid)
        except Exception as e:
            logger.error(
                f"Failed to index transactions in semantic memory: {e}")

        logger.info(
            f"Successfully processed, categorized, and stored {len(new_txs)} new transactions for account {account_id}")

    def get_all_accounts(self):
        all_accounts = []
        provider_logos = {}
        provider_names = {}
        try:
            providers_res = requests.get(
                "https://auth.truelayer.com/api/providers")

            if providers_res is not None and providers_res.status_code == 200:
                for p in providers_res.json():
                    provider_logos[p.get("provider_id")] = p.get("logo_url")
                    provider_names[p.get("provider_id")] = p.get(
                        "display_name")
        except Exception as e:
            pass

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
                    logo_url = provider_logos.get(b.truelayer_provider_id, "")
                    display_name = provider_names.get(
                        b.truelayer_provider_id, b.bank_name)
                    if b.consent_status == 'revoked':
                        all_accounts.append({
                            "account_id": b.truelayer_provider_id,
                            "bank_name": display_name,
                            "provider_name": display_name,
                            "account_number": "****",
                            "sort_code": "00-00-00",
                            "currency": "GBP",
                            "balance": 0.0,
                            "status": "revoked",
                            "provider_id": b.truelayer_provider_id,
                            "logo_url": logo_url
                        })
                        continue

                    for acc in b.accounts:
                        all_accounts.append({
                            "account_id": acc.account_id,
                            "bank_name": display_name,
                            "provider_name": display_name,
                            "account_number": acc.account_number,
                            "sort_code": acc.sort_code or "",
                            "currency": "GBP",
                            "balance": acc.account_balance or 0.0,
                            "status": "active",
                            "provider_id": b.truelayer_provider_id,
                            "logo_url": logo_url
                        })

            return all_accounts
        except Exception:
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
            return pd.DataFrame()

    def get_bank_transactions(self, bank_name_or_id: str, user_uuid: str, from_date: str = None, to_date: str = None):
        try:
            transactions = self.get_transactions(
                bank_name_or_id, user_uuid, start_date=from_date, end_date=to_date)

            needs_api_fetch = True

            if transactions is not None and not transactions.empty:
                temp_dates = pd.to_datetime(
                    transactions['date'], errors='coerce', utc=True)
                oldest_db_record = temp_dates.min()
                if from_date:
                    requested_start = pd.to_datetime(from_date, utc=True)
                    if oldest_db_record <= requested_start + pd.Timedelta(days=5):
                        needs_api_fetch = False
                else:
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
                    return pd.DataFrame()

            all_txs = []

            for row in rows:
                account_id = row[0]
                access_token = self.cipher_suite.decrypt(row[1]).decode()
                provider_id = row[2]

                url = self.base_url + f"/{account_id}/transactions"
                params = {"from": from_date, "to": to_date}

                result = self._make_request(
                    url=url, token=access_token, provider_id=provider_id, params=params)

                if result and result.status_code == 200:
                    data = result.json()
                    transactions_list = data.get(
                        'results', data) if isinstance(data, dict) else data

                    # Consistently store new transactions
                    self._process_and_store_transactions(
                        session, transactions_list, user_uuid, row[2], account_id)

                    all_txs.extend(transactions_list)

            # Re-fetch from DB to ensure we return the categorized, deduplicated records
            return self.get_transactions(bank_name_or_id, user_uuid, start_date=from_date, end_date=to_date)

        except Exception as e:
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
            return False
