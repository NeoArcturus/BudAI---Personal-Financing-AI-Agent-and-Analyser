import pandas as pd
import os
import sys
import json
import hashlib
import logging
from datetime import datetime
from diskcache import Cache
from sqlalchemy import text
import asyncio

from config import SessionLocal
from services.api_integrator.get_account_detail import UserAccounts
from services.logger_setup import get_core_logger
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional

logger = get_core_logger(__name__)

class CategorizedTransaction(BaseModel):
    transaction_uuid: str
    category: str = Field(description="Must exactly match a top-level key in budai_category_rules.json")
    sub_category: Optional[str] = Field(description="Must exactly match a sub-category under the chosen category")

class BatchCategorizationOutput(BaseModel):
    results: List[CategorizedTransaction]

class CategorizerAgent:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache = Cache('./agent_cache')
        rules_path = os.path.join(self.base_dir, "budai_category_rules.json")
        with open(rules_path, "r") as f:
            self.rules = f.read()
            f.seek(0)
            self.valid_categories = list(
                json.load(f)["rules"].keys()) + ["Income", "Uncategorized"]
                
        # The categorizer model runs on port 8001, while the main model runs on 8000
        base_url = os.getenv("CATEGORIZER_LLM_URL", "http://host.docker.internal:8001/v1")
        if not base_url.endswith("/v1"): 
            base_url = f"{base_url}/v1"
            
        self.llm = ChatOpenAI(
            model="qwen3-0.6b-8bit", 
            base_url=base_url, 
            api_key="budai-local", 
            temperature=0,
            max_tokens=2000
        )
        self.structured_llm = self.llm.with_structured_output(BatchCategorizationOutput)
        self._ensure_feedback_table()
        
    def _ensure_feedback_table(self):
        with SessionLocal() as session:
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS transaction_label_feedback (
                    id SERIAL PRIMARY KEY,
                    user_uuid TEXT NOT NULL,
                    transaction_uuid TEXT NOT NULL,
                    corrected_label TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_uuid, transaction_uuid)
                )
            """))
            session.commit()
            
    def save_manual_label(self, user_uuid, transaction_uuid, corrected_label):
        if corrected_label not in self.valid_categories:
            raise ValueError(f"Invalid category label: {corrected_label}")
        self._ensure_feedback_table()
        with SessionLocal() as session:
            session.execute(text("""
                INSERT INTO transaction_label_feedback (user_uuid, transaction_uuid, corrected_label, updated_at)
                VALUES (:user_uuid, :transaction_uuid, :corrected_label, CURRENT_TIMESTAMP)
                ON CONFLICT(user_uuid, transaction_uuid)
                DO UPDATE SET corrected_label = excluded.corrected_label, updated_at = CURRENT_TIMESTAMP
            """), {
                "user_uuid": user_uuid,
                "transaction_uuid": transaction_uuid,
                "corrected_label": corrected_label
            })
            session.execute(text("""
                UPDATE transactions
                SET category = :corrected_label
                WHERE user_uuid = :user_uuid AND transaction_uuid = :transaction_uuid
            """), {
                "user_uuid": user_uuid,
                "transaction_uuid": transaction_uuid,
                "corrected_label": corrected_label
            })
            session.commit()

    def retrain_from_feedback(self, user_uuid):
        return {"trained": True, "reason": "LLM dynamically reads rules, no retraining needed. Feedback applied to DB."}

    def train_global(self):
        return {"trained": True, "reason": "Global LLM rules applied, no local ML training required."}

    async def _categorize_batch(self, batch):
        minimal_batch = [{"id": t["transaction_uuid"], "desc": t["description"], "amount": t["amount"]} for t in batch]
        system_prompt = f"""You are a strict financial categorizer.
You must classify transactions based ONLY on the following rules JSON:
{self.rules}

Output valid JSON matching the exact schema provided. Ensure category strictly matches top-level keys."""
        try:
            response = await self.structured_llm.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(minimal_batch)}
            ])
            return response.results
        except Exception as e:
            logger.error(f"Error in LLM categorization batch: {e}")
            return []

    def execute_cycle(self, identifier, user_uuid, start_date, end_date):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.async_execute_cycle(identifier, user_uuid, start_date, end_date))

    async def async_execute_cycle(self, identifier, user_uuid, start_date, end_date):
        try:
            if str(identifier).upper() == "ALL" or "," in str(identifier):
                raise ValueError("CategorizerAgent strictly handles a single account identifier.")
                
            user_acc = UserAccounts(user_id=user_uuid)
            raw_df = user_acc.get_bank_transactions(identifier, user_uuid, start_date, end_date)
            if raw_df is None or raw_df.empty:
                return None
                
            with SessionLocal() as session:
                feedback_rows = session.execute(text("""
                    SELECT transaction_uuid, corrected_label
                    FROM transaction_label_feedback
                    WHERE user_uuid = :user_uuid
                """), {"user_uuid": user_uuid}).fetchall()
            feedback_map = {row[0]: row[1] for row in feedback_rows}
            
            transactions = []
            for _, row in raw_df.iterrows():
                tx_uuid = row.get('transaction_uuid') or row.get('transaction_id')
                if not tx_uuid:
                    tx_uuid = hashlib.sha256(f"{user_uuid}_{identifier}_{row.get('date')}_{row.get('amount')}_{row.get('description')}".encode()).hexdigest()
                transactions.append({
                    "transaction_uuid": tx_uuid,
                    "description": row.get('description', ''),
                    "amount": row.get('amount', 0.0),
                    "original_row": row.to_dict()
                })
                
            batch_size = 20
            tasks = []
            for i in range(0, len(transactions), batch_size):
                batch = transactions[i:i + batch_size]
                tasks.append(self._categorize_batch(batch))
                
            results = await asyncio.gather(*tasks)
            categorized_list = [item for sublist in results for item in sublist]
            cat_map = {item.transaction_uuid: (item.category, item.sub_category) for item in categorized_list}
            
            final_rows = []
            for tx in transactions:
                tx_uuid = tx["transaction_uuid"]
                orig = tx["original_row"]
                if tx_uuid in feedback_map:
                    category = feedback_map[tx_uuid]
                    sub_category = None
                else:
                    category, sub_category = cat_map.get(tx_uuid, ("Uncategorized", None))
                orig['Category'] = category
                orig['Sub_Category'] = sub_category
                orig['transaction_id'] = tx_uuid
                final_rows.append(orig)
                
            final_df = pd.DataFrame(final_rows)
            with SessionLocal() as session:
                row = session.execute(text("""
                    SELECT a.account_id
                    FROM accounts a
                    JOIN banks b ON a.bank_uuid = b.bank_uuid
                    WHERE (b.bank_name = :identifier OR a.account_id = :identifier) AND a.user_uuid = :user_uuid
                """), {"identifier": identifier, "user_uuid": user_uuid}).fetchone()
                actual_acc_id = row[0] if row else identifier
                
            self._update_sql_memory(final_df, actual_acc_id, user_uuid)
            return final_df
            
        except Exception as e:
            logger.error("An error occurred in async_execute_cycle", exc_info=True)
            logger.error(e)

    def _update_sql_memory(self, df, account_id, user_uuid):
        from models.database_models import Transaction, Bank, Account
        from config import SessionLocal
        with SessionLocal() as session:
            bank = session.query(Bank).join(Account).filter(
                Account.account_id == account_id, Account.user_uuid == user_uuid).first()
            bank_uuid = bank.bank_uuid if bank else None
            df = df.loc[:, ~df.columns.duplicated()].copy()
            records = df.fillna("").to_dict(orient="records")
            seen_ids = set()
            for r in records:
                acc_id_val = r.get('account_id') or account_id
                raw_date = r.get('Date') or r.get('date')
                if raw_date:
                    try:
                        date_val = pd.to_datetime(raw_date, format='ISO8601').to_pydatetime()
                    except Exception:
                        date_val = datetime.now()
                else:
                    date_val = datetime.now()
                amt_raw = r.get('Amount') or r.get('amount') or 0.0
                try:
                    amt_val = float(amt_raw)
                except ValueError:
                    amt_val = 0.0
                desc_val = r.get('Description') or r.get('description') or ''
                cat_val = r.get('Category') or 'Uncategorized'
                sub_cat_val = r.get('Sub_Category')
                tx_hash = hashlib.sha256(
                    f"{user_uuid}_{account_id}_{date_val.strftime('%Y-%m-%d')}_{amt_val}_{desc_val}".encode()).hexdigest()
                tx_id = str(r.get('transaction_id') or r.get('transaction_uuid') or tx_hash)

                if tx_id in seen_ids:
                    continue
                seen_ids.add(tx_id)
                existing = session.query(Transaction).filter_by(
                    transaction_uuid=tx_id, user_uuid=user_uuid).first()
                if existing:
                    existing.category = cat_val
                    existing.sub_category = sub_cat_val
                else:
                    new_tx = Transaction(
                        transaction_uuid=tx_id,
                        user_uuid=user_uuid,
                        bank_uuid=bank_uuid,
                        account_id=acc_id_val,
                        date=date_val,
                        amount=amt_val,
                        description=desc_val,
                        category=cat_val,
                        sub_category=sub_cat_val
                    )
                    session.add(new_tx)
            session.commit()
