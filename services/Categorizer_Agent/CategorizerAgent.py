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
import re

from config import SessionLocal
from services.logger_setup import get_core_logger
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional

logger = get_core_logger(__name__)

class CategorizedTransaction(BaseModel):
    id: int = Field(description="The integer id provided in the input batch")
    category: str = Field(description="Must exactly match one of the main categories: Income, Housing, Food & Dining, Transportation, Utilities, Entertainment & Lifestyle, Subscriptions & Digital Services, Shopping & Retail, Healthcare, Transfers & Payments, Fees & Charges, Savings & Investments, Taxes & Government Payments, Uncategorized")
    sub_category: Optional[str] = Field(description="A short 1-3 word specific sub-category generated dynamically based on the transaction description (e.g. 'Groceries', 'Coffee', 'Train Ticket')")
    tags: Optional[List[str]] = Field(default=[], description="Behavioral, intent, and contextual tags (e.g. '#essential', '#discretionary', '#impulse-buy', '#recurring', '#work-expense')")

class CategorizedResult:
    def __init__(self, transaction_uuid, category, sub_category, tags=None):
        self.transaction_uuid = transaction_uuid
        self.category = category
        self.sub_category = sub_category
        self.tags = tags or []

class BatchCategorizationOutput(BaseModel):
    results: List[CategorizedTransaction]

class CategorizerAgent:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache = Cache('./agent_cache')
        self.valid_categories = [
            "Income",
            "Housing",
            "Food & Dining",
            "Transportation",
            "Utilities",
            "Entertainment & Lifestyle",
            "Subscriptions & Digital Services",
            "Shopping & Retail",
            "Healthcare",
            "Transfers & Payments",
            "Fees & Charges",
            "Savings & Investments",
            "Taxes & Government Payments",
            "Uncategorized"
        ]
        
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
        if not base_url.endswith("/v1"): 
            base_url = f"{base_url}/v1"
            
        self.llm = ChatOpenAI(
            model="lmstudio-community/Qwen3.5-9B-GGUF", # Mac: model="mlx-community/Qwen3.5-4B-4bit", 
            base_url=base_url, 
            api_key="budai-local", 
            temperature=0,
            max_tokens=20000
        )
        self.structured_llm = self.llm.with_structured_output(BatchCategorizationOutput)
        self._ensure_feedback_table()
        
    def _ensure_feedback_table(self):
        pass
            
    def save_manual_label(self, user_uuid, transaction_uuid, corrected_label):
        if corrected_label not in self.valid_categories:
            raise ValueError(f"Invalid category label: {corrected_label}")
        with SessionLocal() as session:
            from models.database_models import Transaction, MerchantRule
            tx = session.query(Transaction).filter_by(transaction_uuid=transaction_uuid).first()
            if not tx: return
            
            merchant = tx.semi_cleaned_description or tx.description
            if merchant:
                merchant = merchant.strip().lower()
                existing_rule = session.query(MerchantRule).filter_by(user_uuid=user_uuid, merchant_name=merchant).first()
                if existing_rule:
                    existing_rule.category = corrected_label
                else:
                    rule = MerchantRule(user_uuid=user_uuid, merchant_name=merchant, category=corrected_label)
                    session.add(rule)
            
            tx.category = corrected_label
            session.commit()

    def retrain_from_feedback(self, user_uuid):
        return {"trained": True, "reason": "Feedback applied to merchant rules in DB."}

    def train_global(self):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.async_train_global())

    def _apply_deterministic_rules(self, tx_dict):
        """Returns (category, sub_category, tags) if deterministic, else None"""
        desc = str(tx_dict.get("description", "")).lower()
        amount = float(tx_dict.get("amount", 0.0))
        date_str = str(tx_dict.get("date", ""))
        
        tags = []
        if amount < -500:
            tags.append("#large-purchase")
            
        try:
            dt = pd.to_datetime(date_str)
            if dt.weekday() >= 5:
                tags.append("#weekend")
            if dt.hour >= 23 or dt.hour < 5:
                tags.append("#late-night")
        except:
            pass

        cat = None
        sub_cat = None
        
        if amount > 0:
            if "savings" in desc or "to " in desc or "from " in desc:
                cat = "Transfers & Payments"
                sub_cat = "Internal Transfer"
        else:
            if "savings" in desc or "to " in desc or "from " in desc:
                cat = "Savings & Investments" if "savings" in desc else "Transfers & Payments"
                sub_cat = "Transfer"
            elif any(x in desc for x in ["spotify", "netflix", "prime", "hulu"]):
                cat = "Subscriptions & Digital Services"
                sub_cat = "Streaming"
                tags.append("#entertainment")
                
        if cat:
            return cat, sub_cat, tags
        return None, None, tags

    async def async_train_global(self):
        try:
            from models.database_models import Transaction, MerchantRule
            
            with SessionLocal() as session:
                uncategorized_txs = session.query(Transaction).filter(
                    (Transaction.category == 'Uncategorized') | 
                    (Transaction.category == None) |
                    (Transaction.category == '') |
                    (Transaction.category == 'Food & Dining') # Re-evaluate to shift to new categories if needed, but lets stick to uncat for now
                ).filter(
                    (Transaction.category == 'Uncategorized') | 
                    (Transaction.category == None) |
                    (Transaction.category == '') |
                    (Transaction.sub_category == None) |
                    (Transaction.sub_category == '')
                ).all()
                
                if not uncategorized_txs:
                    return {"trained": True, "samples": 0, "reason": "All transactions are already categorized."}
                
                # Fetch memory rules
                user_uuids = list(set([tx.user_uuid for tx in uncategorized_txs if tx.user_uuid]))
                rules = session.query(MerchantRule).filter(MerchantRule.user_uuid.in_(user_uuids)).all()
                rule_map = {f"{r.user_uuid}_{r.merchant_name}": r.category for r in rules}
                
                transactions = []
                for tx in uncategorized_txs:
                    transactions.append({
                        "transaction_uuid": tx.transaction_uuid,
                        "user_uuid": tx.user_uuid,
                        "date": str(tx.date.date()) if tx.date else "",
                        "raw_string": tx.description or '',
                        "description": tx.description or '',
                        "semi_cleaned_string": tx.semi_cleaned_description or '',
                        "fully_cleaned_string": tx.fully_cleaned_description or '',
                        "amount": tx.amount or 0.0,
                    })

            logger.info(json.dumps({"message": f"Background Categorizer found {len(transactions)} uncategorized transactions. Processing...", "status_code": 200}))

            batch_size = 100
            semaphore = asyncio.Semaphore(1)
            
            async def sem_task(batch):
                async with semaphore:
                    # Pre-filter with deterministic rules & memory
                    llm_batch = []
                    results = []
                    for tx in batch:
                        cat, sub_cat, d_tags = self._apply_deterministic_rules(tx)
                        
                        merchant_key = f"{tx['user_uuid']}_{str(tx['semi_cleaned_string'] or tx['raw_string']).strip().lower()}"
                        if not cat and merchant_key in rule_map:
                            cat = rule_map[merchant_key]
                            sub_cat = "Manual Rule"
                            
                        if cat:
                            results.append(CategorizedResult(tx['transaction_uuid'], cat, sub_cat, d_tags))
                        else:
                            # Pass to LLM
                            tx['tags_pre'] = d_tags
                            llm_batch.append(tx)
                            
                    if llm_batch:
                        llm_results = await self._categorize_batch(llm_batch)
                        # merge tags
                        for res in llm_results:
                            orig = next((x for x in llm_batch if x['transaction_uuid'] == res.transaction_uuid), None)
                            if orig and orig.get('tags_pre'):
                                res.tags = list(set(res.tags + orig['tags_pre']))
                        results.extend(llm_results)
                        await asyncio.sleep(2.0)
                    return results
            
            tasks = []
            for i in range(0, len(transactions), batch_size):
                batch = transactions[i:i + batch_size]
                tasks.append(sem_task(batch))
                
            total_batches = len(tasks)
            results = []
            for idx, coro in enumerate(asyncio.as_completed(tasks), 1):
                res = await coro
                results.append(res)
                logger.info(json.dumps({"message": f"Categorization Progress: {idx}/{total_batches} batches complete.", "status_code": 200}))
                
            categorized_list = [item for sublist in results for item in sublist]
            
            with SessionLocal() as session:
                updated_count = 0
                for item in categorized_list:
                    existing = session.query(Transaction).filter_by(transaction_uuid=item.transaction_uuid).first()
                    if existing:
                        existing.category = item.category
                        existing.sub_category = item.sub_category
                        existing.tags = item.tags
                        updated_count += 1
                session.commit()

            return {"trained": True, "samples": updated_count, "reason": "Successfully categorized."}

        except Exception as e:
            logger.error(json.dumps({"message": f"Error in background global categorization: {e}", "status_code": 500}))
            return {"trained": False, "samples": 0, "reason": str(e)}

    async def _categorize_batch(self, batch):
        minimal_batch = [{"id": i, "date": t.get("date", ""), "amount": t.get("amount", 0.0), "raw_string": t.get("raw_string", ""), "semi_cleaned_string": t.get("semi_cleaned_string", ""), "fully_cleaned_string": t.get("fully_cleaned_string", "")} for i, t in enumerate(batch)]
        categories_str = ", ".join(self.valid_categories)
        system_prompt = f"""You are a highly intelligent financial categorizer.
Classify each transaction based on your understanding of the transaction strings, amount, and date.
Rely primarily on the `semi_cleaned_string` to deduce the merchant. 
CRITICAL RULE: DO NOT categorize internal bank transfers or 'From Savings' as 'Income'. 'Income' is ONLY for salary, refunds, or external payments. Transfers must be 'Transfers & Payments'.

Output valid JSON matching the exact schema provided. Think step-by-step to deduce intent before categorizing.
- Ensure 'category' strictly matches one of the following main categories: {categories_str}.
- Generate a concise 1-3 word string for 'sub_category' (e.g. 'Groceries', 'Coffee').
- Generate an array of string tags representing behavior and intent (e.g., '#essential', '#discretionary', '#impulse-buy', '#work-expense')."""
        try:
            def _locked_call():
                from services.llm_manager import GlobalLLMManager
                GlobalLLMManager.acquire_background()
                try:
                    return self.structured_llm.invoke([
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(minimal_batch)}
                    ], config={"callbacks": []})
                finally:
                    GlobalLLMManager.release()
            
            response = await asyncio.to_thread(_locked_call)
            
            final_results = []
            for item in response.results:
                idx = item.id
                if 0 <= idx < len(batch):
                    final_results.append(CategorizedResult(
                        transaction_uuid=batch[idx]["transaction_uuid"],
                        category=item.category,
                        sub_category=item.sub_category,
                        tags=item.tags
                    ))
            return final_results
        except Exception as e:
            logger.error(json.dumps({"message": f"Error in LLM categorization batch: {e}", "status_code": 500}))
            return []

    def execute_cycle(self, account_id, user_uuid, start_date, end_date):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.async_execute_cycle(account_id, user_uuid, start_date, end_date))

    async def async_execute_cycle(self, account_id, user_uuid, start_date, end_date):
        try:
            if not account_id or "," in str(account_id):
                raise ValueError("CategorizerAgent strictly handles a single account identifier.")
                
            from services.api_integrator.account_reader import AccountReader
            from models.database_models import MerchantRule
            
            user_acc = AccountReader(user_id=user_uuid)
            raw_df = user_acc.get_transactions(account_id, user_uuid, start_date, end_date)
            if raw_df is None or raw_df.empty:
                return None
                
            with SessionLocal() as session:
                rules = session.query(MerchantRule).filter(MerchantRule.user_uuid == user_uuid).all()
                rule_map = {r.merchant_name: r.category for r in rules}
            
            transactions = []
            for _, row in raw_df.iterrows():
                tx_uuid = row.get('transaction_uuid') or row.get('transaction_id')
                if not tx_uuid:
                    tx_uuid = hashlib.sha256(f"{user_uuid}_{account_id}_{row.get('date')}_{row.get('amount')}_{row.get('description')}".encode()).hexdigest()
                transactions.append({
                    "transaction_uuid": tx_uuid,
                    "description": row.get('description', ''),
                    "amount": row.get('amount', 0.0),
                    "date": row.get('date', ''),
                    "raw_string": row.get('description', ''),
                    "semi_cleaned_string": row.get('semi_cleaned_description', ''),
                    "original_row": row.to_dict()
                })
                
            batch_size = 100
            semaphore = asyncio.Semaphore(1)
            
            async def sem_task(batch):
                async with semaphore:
                    llm_batch = []
                    results = []
                    for tx in batch:
                        cat, sub_cat, d_tags = self._apply_deterministic_rules(tx)
                        
                        merchant_key = str(tx['semi_cleaned_string'] or tx['raw_string']).strip().lower()
                        if not cat and merchant_key in rule_map:
                            cat = rule_map[merchant_key]
                            sub_cat = "Manual Rule"
                            
                        if cat:
                            results.append(CategorizedResult(tx['transaction_uuid'], cat, sub_cat, d_tags))
                        else:
                            tx['tags_pre'] = d_tags
                            llm_batch.append(tx)
                            
                    if llm_batch:
                        llm_results = await self._categorize_batch(llm_batch)
                        for res in llm_results:
                            orig = next((x for x in llm_batch if x['transaction_uuid'] == res.transaction_uuid), None)
                            if orig and orig.get('tags_pre'):
                                res.tags = list(set(res.tags + orig['tags_pre']))
                        results.extend(llm_results)
                        await asyncio.sleep(2.0)
                    return results
                    
            tasks = []
            for i in range(0, len(transactions), batch_size):
                batch = transactions[i:i + batch_size]
                tasks.append(sem_task(batch))
                
            total_batches = len(tasks)
            results = []
            for idx, coro in enumerate(asyncio.as_completed(tasks), 1):
                res = await coro
                results.append(res)
                logger.info(json.dumps({"message": f"Dynamic Categorization Progress: {idx}/{total_batches} batches complete.", "status_code": 200}))
                
            categorized_list = [item for sublist in results for item in sublist]
            cat_map = {item.transaction_uuid: (item.category, item.sub_category, item.tags) for item in categorized_list}
            
            final_rows = []
            for tx in transactions:
                tx_uuid = tx["transaction_uuid"]
                orig = tx["original_row"]
                category, sub_category, tags = cat_map.get(tx_uuid, ("Uncategorized", None, []))
                orig['Category'] = category
                orig['Sub_Category'] = sub_category
                orig['Tags'] = tags
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
                actual_acc_id = row[0] if row else account_id
                
            self._update_sql_memory(final_df, actual_acc_id, user_uuid)
            return final_df
            
        except Exception as e:
            logger.error(json.dumps({"message": f"An error occurred in async_execute_cycle", "status_code": 500}), exc_info=True)
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
                tags_val = r.get('Tags', [])
                
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
                    existing.tags = tags_val
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
                        sub_category=sub_cat_val,
                        tags=tags_val
                    )
                    session.add(new_tx)
            session.commit()
