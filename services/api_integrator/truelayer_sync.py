import json
import requests
import time
import hashlib
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

class TrueLayerSync:
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
                headers["Authorization"] = f"Bearer {new_token}"
                res = requests.get(url, headers=headers, params=params)
        if res is not None and res.status_code == 403:
            err_data = res.json()
            logger.error(json.dumps({"message": f"Access forbidden (403) for {url}: {err_data}", "status_code": 500}))
            if isinstance(err_data, dict) and (err_data.get("error") == "sca_exceeded" or "PSU" in str(err_data)):
                raise PermissionError("SECURITY LOCK")
        if res is not None:
            if res.status_code == 404 and url.endswith("/balance"):
                pass # Suppress noisy 404 logs for unsupported balance endpoints
            else:
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
                    logger.warning(json.dumps({"message": f"Bank connection not found: {bank_uuid}", "status_code": 400}))
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
                    logger.debug(
                        f"Found {len(results)} accounts for provider {provider_id}")
                    if not results:
                        return False
                    for acc_det in results:
                        acc_id = acc_det.get("account_id")
                        display_name = acc_det.get(
                            "display_name", "Unknown Account")
                        account_number_info = acc_det.get("account_number")
                        if isinstance(account_number_info, list) and len(account_number_info) > 0:
                            account_number_info = account_number_info[0]
                        if not isinstance(account_number_info, dict):
                            account_number_info = {}
                        sort_code = str(
                            account_number_info.get("sort_code", ""))
                        acc_no = str(account_number_info.get("number", ""))
                        
                        if not acc_no and not sort_code:
                            for sibling in results:
                                sib_info = sibling.get("account_number")
                                if isinstance(sib_info, list) and len(sib_info) > 0:
                                    sib_info = sib_info[0]
                                if isinstance(sib_info, dict):
                                    s_acc = str(sib_info.get("number", ""))
                                    s_sort = str(sib_info.get("sort_code", ""))
                                    if s_acc or s_sort:
                                        acc_no = s_acc
                                        sort_code = s_sort
                                        break
                        acc_balance = 0.0
                        bal_res = self._make_request(
                            f"{self.base_url}/{acc_id}/balance", access_token, provider_id)
                        if bal_res is not None:
                            if bal_res.status_code == 200:
                                bal_data = bal_res.json().get("results", [{}])[0]
                                acc_balance = bal_data.get("available", bal_data.get("current", 0.0))
                            elif bal_res.status_code == 404:
                                logger.debug(json.dumps({"message": f"Balance not supported for account {acc_id} (404). Defaulting to 0.0", "status_code": 100}))
                        from sqlalchemy.dialects.postgresql import insert as pg_insert

                        stmt = pg_insert(Account).values(
                            account_id=acc_id,
                            user_uuid=user_uuid,
                            bank_uuid=bank_uuid,
                            account_number=acc_no,
                            sort_code=sort_code,
                            account_balance=float(acc_balance),
                            currency=acc_det.get("currency", "GBP"),
                            account_type=acc_det.get("account_type", "TRANSACTION"),
                            display_name=display_name
                        )

                        update_dict = {
                            c.name: c
                            for c in stmt.excluded
                            if c.name not in ["account_id", "last_synced_at"]
                        }

                        stmt = stmt.on_conflict_do_update(
                            index_elements=["account_id"],
                            set_=update_dict
                        )
                        
                        session.execute(stmt)
                        session.commit()
                        tx_url = f"{self.base_url}/{acc_id}/transactions"
                        
                        target_acc = session.query(Account).filter_by(account_id=acc_id).first()
                        if target_acc and target_acc.last_synced_at:
                            from_date = (target_acc.last_synced_at - timedelta(days=3)).strftime('%Y-%m-%d')
                        else:
                            from_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
                            
                        to_date = datetime.now().strftime('%Y-%m-%d')
                        import os
                        webhook_base = os.getenv("WEBHOOK_BASE_URL", "https://api.budai.app")
                        webhook_url = f"{webhook_base}/api/webhooks/truelayer?user_uuid={user_uuid}&bank_uuid={bank_uuid}&acc_id={acc_id}"
                        
                        tx_params = {
                            "from": from_date, 
                            "to": to_date, 
                            "async": "true", 
                            "webhook_uri": webhook_url
                        }
                        
                        tx_res = self._make_request(
                            tx_url, access_token, provider_id, params=tx_params)
                            
                        if tx_res is not None and tx_res.status_code in [200, 202]:
                            res_json = tx_res.json()
                    
                    logger.info(
                        f"Finished initialization for bank connection: {bank_uuid}")
                    try:
                        import asyncio
                        from utils.cache_utils import clear_user_cache
                        async def _clear_cache():
                            clear_user_cache(str(user_uuid), namespace="accounts")
                            clear_user_cache(str(user_uuid), namespace="transactions")
                            clear_user_cache(str(user_uuid), namespace="categorizer")
                        
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(_clear_cache())
                        except RuntimeError:
                            asyncio.run(_clear_cache())
                    except Exception as ce:
                        logger.error(json.dumps({"message": f"Failed to clear cache: {ce}", "status_code": 500}))
                    return True
                return False
        except Exception as e:
            logger.error(json.dumps({"message": f"Error initializing accounts: {e}", "status_code": 500}), exc_info=True)
            return False

    def trigger_sync(self, account_id: str, user_uuid: str, from_date: str = None, to_date: str = None):
        try:
            with SessionLocal() as session:
                row = session.query(Account.account_id, Bank.access_token, Bank.bank_uuid, Bank.truelayer_provider_id, Account.last_synced_at)\
                             .join(Bank)\
                             .filter(Account.account_id == account_id, Account.user_uuid == user_uuid).first()
                if not row:
                    return

            acc_id, enc_token, b_uuid, p_id, last_synced = row
                
            is_deep_sync = False
            if from_date:
                try:
                    req_start = pd.to_datetime(from_date, utc=True)
                    with SessionLocal() as s2:
                        oldest = s2.query(Transaction.date).filter_by(account_id=acc_id).order_by(Transaction.date.asc()).first()
                        if not oldest or oldest[0].replace(tzinfo=None) > req_start.replace(tzinfo=None) + timedelta(days=3):
                            is_deep_sync = True
                except:
                    pass
            
            if not is_deep_sync and last_synced and (datetime.utcnow() - last_synced).total_seconds() < 3600:
                return
                
            try:
                access_token = self.cipher_suite.decrypt(bytes(enc_token)).decode()
                url = f"{self.base_url}/{acc_id}/transactions"
                params = {}
                
                if is_deep_sync:
                    params["from"] = pd.to_datetime(from_date, utc=True).strftime('%Y-%m-%d')
                    if to_date:
                        params["to"] = pd.to_datetime(to_date, utc=True).strftime('%Y-%m-%d')
                elif last_synced:
                    overlap_start = last_synced - pd.Timedelta(days=3)
                    params["from"] = overlap_start.strftime('%Y-%m-%d')
                else:
                    pass
                import os
                webhook_base = os.getenv("WEBHOOK_BASE_URL", "https://api.budai.app")
                params["async"] = "true"
                params["webhook_uri"] = f"{webhook_base}/api/webhooks/truelayer?user_uuid={user_uuid}&bank_uuid={b_uuid}&acc_id={acc_id}"

                res = self._make_request(url, access_token, p_id, params=params)
            except Exception as e:
                logger.error(json.dumps({"message": f"Failed to fetch API transactions for {acc_id}: {e}", "status_code": 500}))
        except Exception:
            logger.error(json.dumps({"message": f"Error in trigger_sync", "status_code": 500}), exc_info=True)
        
    def process_and_store_transactions(self, session, tx_data, user_uuid, bank_uuid, account_id):
        from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
        if not tx_data:
            return
        new_txs = []
        seen_in_batch = set()
        for tx in tx_data:
            norm_id = tx.get("normalised_provider_transaction_id")
            raw_tx_id = tx.get("transaction_id")
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
            currency = str(tx.get("currency", "GBP"))
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
            
            tx_id = norm_id or raw_tx_id or tx_hash
            
            if tx_id in seen_in_batch or tx_hash in seen_in_batch:
                continue
            import re
            
            # --- ETL Engine: Tri-Context Transformation ---
            raw_string = desc_val
            
            # 1. Semi-Cleaned (Stop-Word Removal)
            stop_words = ["card transaction of", "gbp", "issued by", "london", "contactless", "direct debit", "standing order", "visa", "mastercard"]
            semi_cleaned = raw_string
            for word in stop_words:
                semi_cleaned = re.sub(r'(?i)\b' + re.escape(word) + r'\b', '', semi_cleaned)
            # Remove consecutive whitespace left over by stop word removal
            semi_cleaned = re.sub(r'\s+', ' ', semi_cleaned).strip()
            
            # 2. Fully Cleaned (Aggressive Regex)
            fully_cleaned = re.sub(r'[^a-zA-Z\s]', '', raw_string).strip().lower()
            fully_cleaned = re.sub(r'\s+', ' ', fully_cleaned)

            new_txs.append({
                "transaction_uuid": tx_id,
                "user_uuid": user_uuid,
                "bank_uuid": bank_uuid,
                "account_id": account_id,
                "provider_transaction_id": tx_id,
                "date": date_val,
                "amount": amount,
                "currency": currency,
                "description": raw_string,
                "semi_cleaned_description": semi_cleaned,
                "fully_cleaned_description": fully_cleaned,
                "category": "Uncategorised"
            })
            seen_in_batch.add(tx_id)
            seen_in_batch.add(tx_hash)
            
        if not new_txs:
            return
            
        # --- PHASE 2: Fast Path RAG Intercept ---
        import os
        from langchain_openai import OpenAIEmbeddings
        from models.database_models import MerchantKnowledge
        from sqlalchemy import text
        
        try:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
            if not base_url.endswith("/v1"): 
                base_url = f"{base_url}/v1"
                
            embeddings_model = OpenAIEmbeddings(
                base_url=base_url,
                model="text-embedding-nomic-embed-text-v1.5",
                api_key="budai-local",
                check_embedding_ctx_length=False
            )
            
            # Extract strings to embed (using semi_cleaned as it strikes a good balance)
            strings_to_embed = [tx["semi_cleaned_description"] for tx in new_txs]
            if strings_to_embed:
                vectors = embeddings_model.embed_documents(strings_to_embed)
                
                for idx, tx in enumerate(new_txs):
                    v = vectors[idx]
                    # Cosine distance < 0.05
                    query_with_dist = text("""
                        SELECT category, clean_merchant_name, (embedding <=> :vec) as distance
                        FROM merchant_knowledge 
                        ORDER BY embedding <=> :vec 
                        LIMIT 1
                    """)
                    result_dist = session.execute(query_with_dist, {"vec": str(v)}).first()
                    if result_dist and result_dist.distance < 0.05:
                        tx["category"] = result_dist.category
        except Exception as e:
            from services.logger_setup import get_core_logger
            logger = get_core_logger(__name__)
            logger.error(f"RAG Intercept Fast Path failed: {e}")

        from sqlalchemy.dialects.postgresql import insert as pg_insert
        
        stmt = pg_insert(Transaction).values(new_txs)
        
        update_dict = {
            "date": stmt.excluded.date,
            "amount": stmt.excluded.amount,
            "currency": stmt.excluded.currency,
            "description": stmt.excluded.description,
            "semi_cleaned_description": stmt.excluded.semi_cleaned_description,
            "fully_cleaned_description": stmt.excluded.fully_cleaned_description
        }
        
        stmt = stmt.on_conflict_do_update(
            constraint="uq_transaction_account_provider",
            set_=update_dict
        )
        
        session.execute(stmt)
        # Update last_synced_at
        session.execute(text("UPDATE accounts SET last_synced_at = :now WHERE account_id = :acc_id"), {"now": datetime.utcnow(), "acc_id": account_id})
        session.commit()

        # Update cache and run ML in a separate thread
        def trigger_categorization():
            try:
                import asyncio
                from services.Categorizer_Agent.lazy_ml import categorize_specific_transactions_bg
                tx_uuids = [tx["transaction_uuid"] for tx in new_txs]
                asyncio.run(categorize_specific_transactions_bg(tx_uuids, user_uuid))
                from utils.cache_utils import clear_user_cache
                clear_user_cache(str(user_uuid), namespace="transactions")
                clear_user_cache(str(user_uuid), namespace="categorizer")
            except Exception as e:
                logger.error(json.dumps({"message": f"Failed in trigger_categorization: {e}", "status_code": 500}), exc_info=True)
            finally:
                from utils.state_manager import clear_account_state
                from utils.cache_utils import clear_user_cache
                
                clear_account_state(account_id)
                clear_user_cache(str(user_uuid), namespace="transactions")
                clear_user_cache(str(user_uuid), namespace="categorizer")

        
        import threading
        threading.Thread(target=trigger_categorization, daemon=True).start()
