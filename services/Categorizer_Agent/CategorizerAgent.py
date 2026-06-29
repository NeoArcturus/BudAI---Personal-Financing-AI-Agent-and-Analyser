import pandas as pd
import os
import sys
import json
import hashlib
import logging
from datetime import datetime
from diskcache import Cache
from sqlalchemy import text
from config import SessionLocal
from services.Categorizer_Agent.training.model_trainer import CategorizerTrainer
from services.Categorizer_Agent.categorizer.preprocessor import Preprocessor
from services.Categorizer_Agent.categorizer.categorizer import Categorizer
from services.api_integrator.get_account_detail import UserAccounts
from services.logger_setup import get_core_logger
logger = get_core_logger(__name__)

sys.path.append(os.path.abspath(os.path.join(

    os.path.dirname(__file__), '..', '..')))
class CategorizerAgent:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(self.base_dir, "saved_model")
        self.enc_dir = os.path.join(self.base_dir, "saved_label_enc")
        self.local_st_path = os.path.join(self.model_dir, "st_model_local")
        self.cache = Cache('./agent_cache')
        self.categorizer = Categorizer()
        rules_path = os.path.join(self.base_dir, "budai_category_rules.json")
        with open(rules_path, "r") as f:
            self.valid_categories = list(
                json.load(f)["rules"].keys()) + ["Income", "Uncategorized"]
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
        self._ensure_feedback_table()
        with SessionLocal() as session:
            rows = session.execute(text("""
                SELECT
                    t.transaction_uuid,
                    t.account_id,
                    t.date,
                    t.amount,
                    t.description,
                    t.category,
                    f.corrected_label
                FROM transactions t
                LEFT JOIN transaction_label_feedback f
                    ON f.transaction_uuid = t.transaction_uuid AND f.user_uuid = t.user_uuid
                WHERE t.user_uuid = :user_uuid
            """), {"user_uuid": user_uuid}).fetchall()
        if not rows:
            return {"trained": False, "reason": "No transactions available for training."}
        records = []
        for row in rows:
            tx_id, acc_id, date_val, amount, description, category, corrected_label = row
            final_label = corrected_label or category or "Uncategorized"
            records.append({
                "transaction_id": tx_id,
                "transaction_uuid": tx_id,
                "account_id": acc_id,
                "timestamp": date_val.isoformat() if hasattr(date_val, "isoformat") else str(date_val),
                "amount": amount,
                "description": description,
                "Category": final_label
            })
        raw_df = pd.DataFrame(records)
        proc = Preprocessor(raw_df, self.local_st_path)
        train_df, embeddings = proc.preprocess_for_inference()
        if "Category" not in train_df.columns:
            return {"trained": False, "reason": "No labels found for training."}
        train_df["Category"] = raw_df["Category"].values
        trainer = CategorizerTrainer(
            train_df, embeddings, self.model_dir, self.enc_dir)
        success = trainer.train()
        return {"trained": success, "samples": len(train_df)}

    def train_global(self):
        xgb_model_path = os.path.join(self.model_dir, "gbm_model.joblib")
        enc_path = os.path.join(self.enc_dir, "label_encoder.joblib")
        if os.path.exists(xgb_model_path) and os.path.exists(enc_path):
            return {"trained": False, "reason": "Model already exists."}
            
        self._ensure_feedback_table()
        with SessionLocal() as session:
            rows = session.execute(text("""
                SELECT
                    t.transaction_uuid,
                    t.account_id,
                    t.date,
                    t.amount,
                    t.description,
                    t.category
                FROM transactions t
            """)).fetchall()
            
        if not rows:
            logger.info("No transactions in DB for global training.")
            return {"trained": False, "reason": "No transactions available for training."}
            
        records = []
        for row in rows:
            tx_id, acc_id, date_val, amount, description, category = row
            records.append({
                "transaction_id": tx_id,
                "transaction_uuid": tx_id,
                "account_id": acc_id,
                "timestamp": date_val.isoformat() if hasattr(date_val, "isoformat") else str(date_val),
                "amount": amount,
                "description": description,
                "Category": category
            })
            
        raw_df = pd.DataFrame(records)
        proc = Preprocessor(raw_df, self.local_st_path)
        train_df, embeddings = proc.preprocess_for_training()
        trainer = CategorizerTrainer(
            train_df, embeddings, self.model_dir, self.enc_dir)
        success = trainer.train()
        return {"trained": success, "samples": len(train_df)}
    def execute_cycle(self, identifier, user_uuid, start_date, end_date):
        try:
            if str(identifier).upper() == "ALL" or "," in str(identifier):
                raise ValueError(
                    "CategorizerAgent strictly handles a single account identifier.")
            user_acc = UserAccounts(user_id=user_uuid)
            raw_df = user_acc.get_bank_transactions(
                identifier, user_uuid, start_date, end_date)
            logger.info("Transactions obtained:")
            logger.info(raw_df)
            if raw_df is None or raw_df.empty:
                return None
            proc = Preprocessor(raw_df, self.local_st_path)
            xgb_model_path = os.path.join(self.model_dir, "gbm_model.joblib")
            enc_path = os.path.join(self.enc_dir, "label_encoder.joblib")
            if not (os.path.exists(xgb_model_path) and os.path.exists(enc_path)):
                training_df, embeddings = proc.preprocess_for_training()
                trainer = CategorizerTrainer(
                    training_df, embeddings, self.model_dir, self.enc_dir)
                trainer.train()
                clean_df = training_df.drop(columns=['Category'])
            else:
                clean_df, embeddings = proc.preprocess_for_inference()
            final_df = self.categorizer.predict(
                clean_df, embeddings, xgb_model_path, enc_path)
            with SessionLocal() as session:
                feedback_rows = session.execute(text("""
                    SELECT transaction_uuid, corrected_label
                    FROM transaction_label_feedback
                    WHERE user_uuid = :user_uuid
                """), {"user_uuid": user_uuid}).fetchall()
            feedback_map = {row[0]: row[1] for row in feedback_rows}
            if "transaction_id" in final_df.columns:
                final_df["Category"] = final_df.apply(
                    lambda r: feedback_map.get(
                        str(r.get("transaction_id")), r.get("Category")),
                    axis=1
                )
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
        except ValueError as ve:
            logger.error("An error occurred in this block", exc_info=True)
            raise ve
        except Exception as e:
            logger.error("An error occurred in this block", exc_info=True)
            logger.error(e)
    def _update_sql_memory(self, df, account_id, user_uuid):
        from models.database_models import Transaction, Bank, Account
        from config import SessionLocal
        import uuid
        from datetime import datetime
        import pandas as pd
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
                        logger.error("An error occurred in this block", exc_info=True)
                        date_val = datetime.now()
                else:
                    date_val = datetime.now()
                amt_raw = r.get('Amount') or r.get('amount') or 0.0
                try:
                    amt_val = float(amt_raw)
                except ValueError:
                    logger.error("An error occurred in this block", exc_info=True)
                    amt_val = 0.0
                desc_val = r.get('Description') or r.get('description') or ''
                cat_val = r.get('Category') or 'Uncategorized'
                tx_hash = hashlib.sha256(
                    f"{user_uuid}_{account_id}_{date_val.strftime('%Y-%m-%d')}_{amt_val}_{desc_val}".encode()).hexdigest()
                tx_id = str(r.get('transaction_id') or r.get(
                    'transaction_uuid') or tx_hash)

                if tx_id in seen_ids:
                    continue
                seen_ids.add(tx_id)

                existing_tx = session.query(Transaction).filter_by(
                    transaction_uuid=tx_id).first()
                if existing_tx:
                    existing_tx.category = cat_val
                else:
                    new_tx = Transaction(
                        transaction_uuid=tx_id,
                        user_uuid=user_uuid,
                        bank_uuid=bank_uuid,
                        account_id=acc_id_val,
                        date=date_val,
                        amount=amt_val,
                        category=cat_val,
                        description=desc_val
                    )
                    session.add(new_tx)
            session.commit()
    def get_classification_report(self):
        report_path = os.path.join(self.model_dir, "classification_report.txt")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                return f.read()
        return "Classification report not found."
