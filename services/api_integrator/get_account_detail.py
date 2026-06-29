import requests
import time
import uuid
import hashlib
import os
import pandas as pd
from typing import Any
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from services.api_integrator.access_token_generator import AccessTokenGenerator
from config import SessionLocal, TRUELAYER_BASE_URL, ENCRYPTION_KEY
from models.database_models import Account, Bank, Transaction
from sqlalchemy import text
from services.logger_setup import get_core_logger
logger = get_core_logger(__name__)


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
                        from_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
                        to_date = datetime.now().strftime('%Y-%m-%d')
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
            logger.error("An error occurred in this block", exc_info=True)
            logger.error(f"Error initializing accounts: {e}", exc_info=True)
            return False

    def _process_and_store_transactions(self, session, tx_data, user_uuid, bank_uuid, account_id):
        from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
        from services.Categorizer_Agent.categorizer.preprocessor import Preprocessor
        if not tx_data:
            logger.info(
                f"No transaction data to process for account {account_id}")
            return
        logger.info(
            f"Processing {len(tx_data)} transactions for account {account_id}")
        new_txs = []
        seen_in_batch = set()
        for tx in tx_data:
            tx_id = tx.get("transaction_id", str(uuid.uuid4()))
            date_str = tx.get("timestamp")
            if date_str:
                try:
                    date_val = datetime.fromisoformat(
                        date_str.replace("Z", "+00:00"))
                except Exception:
                    logger.error(
                        "An error occurred in this block", exc_info=True)
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
            tx_hash = hashlib.sha256(
                f"{user_uuid}_{account_id}_{date_val.strftime('%Y-%m-%d')}_{amount}_{desc_val}".encode()).hexdigest()
            
            if tx_id in seen_in_batch or tx_hash in seen_in_batch:
                continue

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
                seen_in_batch.add(tx_id)
                seen_in_batch.add(tx_hash)
        if not new_txs:
            logger.info(
                f"All {len(tx_data)} transactions for account {account_id} are already in the database. Skipping.")
            return
        logger.info(
            f"Found {len(new_txs)} new transactions to categorize and store for account {account_id}")
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
            category_map = categorized_df.set_index(
                "transaction_uuid")["Category"].to_dict()
            for tx in new_txs:
                tx["category"] = category_map.get(
                    tx["transaction_uuid"], "Uncategorized")
            logger.info("AI Categorization completed.")
        else:
            logger.warning(
                f"Categorization models not found at {xgb_model_path}. Storing as 'Uncategorized'.")
        
        # Fallback regex categorization for any Uncategorized transactions
        rules_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Categorizer_Agent", "budai_category_rules.json")
        try:
            import json
            import re
            with open(rules_path, "r") as f:
                rules = json.load(f).get("rules", {})
            for tx in new_txs:
                if tx.get("category", "Uncategorized") == "Uncategorized":
                    desc_lower = str(tx.get("description", "")).lower()
                    for cat_name, pattern in rules.items():
                        if re.search(pattern, desc_lower):
                            tx["category"] = cat_name
                            break
        except Exception as e:
            logger.error(f"Fallback categorization failed: {e}")
        new_txs.sort(key=lambda x: x["date"])
        logger.info(f"Committing {len(new_txs)} transactions to database...")
        for tx_dict in new_txs:
            new_tx = Transaction(**tx_dict)
            session.add(new_tx)
        session.commit()
        from services.memory_service import MemoryService
        try:
            mem = MemoryService()
            mem.index_transactions(new_txs, user_uuid)
        except Exception as e:
            logger.error("An error occurred in this block", exc_info=True)
            logger.error(
                f"Failed to index transactions in semantic memory: {e}")
        logger.info(
            f"Successfully processed, categorized, and stored {len(new_txs)} new transactions for account {account_id}")

    def get_all_accounts(self, skip_sync=False):
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
            logger.error("An error occurred in this block", exc_info=True)
            pass
        try:
            with SessionLocal() as session:
                banks = session.query(Bank).filter_by(
                    user_uuid=self.user_id).all()
            
            if not skip_sync:
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
            logger.error("An error occurred in this block", exc_info=True)
            return []

    def get_account_balance(self, bank_name_or_id, user_uuid, account_type="TRANSACTION"):
        with SessionLocal() as session:
            acc = session.query(Account).join(Bank).filter(
                (Bank.bank_name.ilike(f"%{bank_name_or_id}%")) | (
                    Account.account_id == bank_name_or_id),
                Account.user_uuid == user_uuid
            ).first()
        if acc and acc.account_balance is not None:
            return float(acc.account_balance)
        return 0.0

    def get_transactions(self, identifier, user_uuid, start_date=None, end_date=None, expense_only=False):
        try:
            with SessionLocal() as session:
                query = session.query(Transaction)
                
                identifiers = [identifier] if isinstance(identifier, str) else identifier
                
                if identifiers and "ALL" not in [str(i).upper() for i in identifiers]:
                    query = query.join(Account).join(Bank).filter(
                        (Bank.bank_name.in_(identifiers)) | (Account.account_id.in_(identifiers))
                    )
                
                query = query.filter(Transaction.user_uuid == user_uuid)

                if start_date:
                    query = query.filter(Transaction.date >= start_date)
                if end_date:
                    query = query.filter(Transaction.date <= end_date)
                if expense_only:
                    query = query.filter(Transaction.amount < 0)
                
                txs = query.order_by(Transaction.date.desc()).all()
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
                        "category": tx.category,
                        "bank_uuid": tx.bank_uuid,
                        "account_id": tx.account_id
                    })
                return pd.DataFrame(transactions)
        except Exception:
            logger.error("Error in get_transactions", exc_info=True)
            return pd.DataFrame()

    def get_bank_transactions(self, identifier: Any, user_uuid: str, from_date: str = None, to_date: str = None, expense_only: bool = False):
        try:
            transactions = self.get_transactions(
                identifier, user_uuid, start_date=from_date, end_date=to_date, expense_only=expense_only)
            
            needs_api_fetch = True
            if not transactions.empty:
                temp_dates = pd.to_datetime(transactions['date'], format='ISO8601', errors='coerce', utc=True)
                if from_date:
                    requested_start = pd.to_datetime(from_date, format='ISO8601', utc=True)
                    if temp_dates.min() <= requested_start + pd.Timedelta(days=3):
                        needs_api_fetch = False
                else:
                    needs_api_fetch = False

            if not needs_api_fetch:
                return transactions

            with SessionLocal() as session:
                identifiers = [identifier] if isinstance(identifier, str) else identifier
                query_base = session.query(Account.account_id, Bank.access_token, Bank.bank_uuid, Bank.truelayer_provider_id).join(Bank)
                
                if identifiers and "ALL" not in [str(i).upper() for i in identifiers]:
                    query_base = query_base.filter(
                        (Bank.bank_name.in_(identifiers)) | (Account.account_id.in_(identifiers))
                    )
                
                rows = query_base.filter(Account.user_uuid == user_uuid).all()
                if not rows:
                    return pd.DataFrame()

            for row in rows:
                acc_id, enc_token, b_uuid, p_id = row
                try:
                    access_token = self.cipher_suite.decrypt(bytes(enc_token)).decode()
                    url = f"{self.base_url}/{acc_id}/transactions"
                    params = {"from": from_date, "to": to_date}
                    res = self._make_request(url, access_token, p_id, params=params)
                    
                    if res and res.status_code == 200:
                        tx_list = res.json().get('results', [])
                        self._process_and_store_transactions(session, tx_list, user_uuid, b_uuid, acc_id)
                except Exception as e:
                    logger.error(f"Failed to fetch API transactions for {acc_id}: {e}")

            return self.get_transactions(identifier, user_uuid, start_date=from_date, end_date=to_date, expense_only=expense_only)
        except Exception:
            logger.error("Error in get_bank_transactions", exc_info=True)
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
            logger.error("An error occurred in this block", exc_info=True)
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
            logger.error("An error occurred in this block", exc_info=True)
            return False
