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
    id: int = Field(description="The integer id provided in the input batch")
    category: str = Field(description="Must exactly match one of the main categories: Food & Dining, Transportation, Bills & Utilities, Shopping, Entertainment, Health & Wellness, Transfers & Investments, High-Risk / Anomaly, Income, Uncategorized")
    sub_category: Optional[str] = Field(description="A short 1-3 word specific sub-category generated dynamically based on the transaction description (e.g. 'Groceries', 'Coffee', 'Train Ticket')")

class CategorizedResult:
    def __init__(self, transaction_uuid, category, sub_category):
        self.transaction_uuid = transaction_uuid
        self.category = category
        self.sub_category = sub_category

class BatchCategorizationOutput(BaseModel):
    results: List[CategorizedTransaction]

class CategorizerAgent:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache = Cache('./agent_cache')
        self.valid_categories = [
            "Food & Dining",
            "Transportation",
            "Bills & Utilities",
            "Shopping",
            "Entertainment",
            "Health & Wellness",
            "Transfers & Investments",
            "High-Risk / Anomaly",
            "Income",
            "Uncategorized"
        ]
        
        # We use the main model on port 8000 for categorization as well
        base_url = os.getenv("LLM_BASE_URL", "http://host.docker.internal:8000/v1")
        if not base_url.endswith("/v1"): 
            base_url = f"{base_url}/v1"
            
        self.llm = ChatOpenAI(
            model="mlx-community/Qwen3.5-4B-4bit", 
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
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.async_train_global())

    async def async_train_global(self):
        try:
            from models.database_models import Transaction
            
            with SessionLocal() as session:
                # Fetch transactions that are uncategorized, missing a category, or missing a sub_category
                uncategorized_txs = session.query(Transaction).filter(
                    (Transaction.category == 'Uncategorized') | 
                    (Transaction.category == None) |
                    (Transaction.category == '') |
                    (Transaction.sub_category == None) |
                    (Transaction.sub_category == '')
                ).all()
                
                if not uncategorized_txs:
                    return {"trained": True, "samples": 0, "reason": "All transactions are already categorized."}
                
                # Convert to dict for processing
                transactions = []
                for tx in uncategorized_txs:
                    transactions.append({
                        "transaction_uuid": tx.transaction_uuid,
                        "date": str(tx.date.date()) if tx.date else "",
                        "description": tx.description or '',
                        "amount": tx.amount or 0.0,
                    })

            logger.info(f"Background Categorizer found {len(transactions)} uncategorized transactions. Processing via LLM...")

            batch_size = 20
            semaphore = asyncio.Semaphore(2)
            
            async def sem_task(batch):
                async with semaphore:
                    return await self._categorize_batch(batch)
            
            tasks = []
            for i in range(0, len(transactions), batch_size):
                batch = transactions[i:i + batch_size]
                tasks.append(sem_task(batch))
                
            results = await asyncio.gather(*tasks)
            categorized_list = [item for sublist in results for item in sublist]
            
            # Update database with new categories
            with SessionLocal() as session:
                updated_count = 0
                for item in categorized_list:
                    existing = session.query(Transaction).filter_by(transaction_uuid=item.transaction_uuid).first()
                    if existing:
                        existing.category = item.category
                        existing.sub_category = item.sub_category
                        updated_count += 1
                session.commit()

            return {"trained": True, "samples": updated_count, "reason": "Successfully categorized via background LLM batching."}

        except Exception as e:
            logger.error(f"Error in background global categorization: {e}")
            return {"trained": False, "samples": 0, "reason": str(e)}

    async def _categorize_batch(self, batch):
        minimal_batch = [{"id": i, "date": t.get("date", ""), "amount": t.get("amount", 0.0), "desc": t.get("description", t.get("desc", ""))} for i, t in enumerate(batch)]
        categories_str = ", ".join(self.valid_categories)
        system_prompt = f"""You are a highly intelligent financial categorizer.
Classify each transaction based on your understanding of the transaction description, amount, and date.

Output valid JSON matching the exact schema provided. 
- Ensure 'category' strictly matches one of the following main categories: {categories_str}.
- Generate a concise 1-3 word string for 'sub_category' that best describes the specific purchase (e.g. 'Groceries', 'Coffee', 'Train Ticket') based on the transaction description."""
        try:
            response = await self.structured_llm.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(minimal_batch)}
            ])
            
            final_results = []
            for item in response.results:
                idx = item.id
                if 0 <= idx < len(batch):
                    final_results.append(CategorizedResult(
                        transaction_uuid=batch[idx]["transaction_uuid"],
                        category=item.category,
                        sub_category=item.sub_category
                    ))
            return final_results
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
            semaphore = asyncio.Semaphore(2)
            
            async def sem_task(batch):
                async with semaphore:
                    return await self._categorize_batch(batch)
                    
            tasks = []
            for i in range(0, len(transactions), batch_size):
                batch = transactions[i:i + batch_size]
                tasks.append(sem_task(batch))
                
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
