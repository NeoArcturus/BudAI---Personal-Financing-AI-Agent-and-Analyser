import sqlite3
import pandas as pd
import os
import sys
import uuid
from diskcache import Cache
from services.Categorizer_Agent.training.model_trainer import CategorizerTrainer
from services.Categorizer_Agent.categorizer.preprocessor import Preprocessor
from services.Categorizer_Agent.categorizer.categorizer import Categorizer
from services.api_integrator.get_account_detail import UserAccount

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))


class CategorizerAgent:
    def __init__(self, db_path="budai_memory.db"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(self.base_dir, "saved_model")
        self.enc_dir = os.path.join(self.base_dir, "saved_label_enc")
        self.local_st_path = os.path.join(self.model_dir, "st_model_local")
        self.db_path = db_path
        self.cache = Cache('./agent_cache')
        self.categorizer = Categorizer()

    def execute_cycle(self, identifier, user_uuid, start_date, end_date):
        try:
            user_acc = UserAccount(identifier, user_uuid, self.db_path)
            if user_acc.needs_user_clarification:
                raise ValueError("MULTIPLE_ACCOUNTS")
            raw_df = user_acc.all_transactions(start_date, end_date)
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

            root_dir = os.path.abspath(os.path.join(self.base_dir, '..', '..'))
            csv_dir = os.path.join(root_dir, "saved_media", "csvs")
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(
                csv_dir, "categorized_data_" + str(user_acc.account_id) + ".csv")

            final_df.to_csv(csv_path, index=False)
            self._update_sql_memory(
                final_df, user_acc.account_id, user_uuid, csv_path)
            return final_df
        except ValueError as ve:
            raise ve
        except Exception as e:
            pass

    def _update_sql_memory(self, df, account_id, user_uuid, csv_path):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for i, row in df.iterrows():
                tx_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT OR REPLACE INTO transactions (transaction_uuid, user_uuid, truelayer_account_id, csv_file_path)
                    VALUES (?, ?, ?, ?)
                """, (tx_id, user_uuid, str(account_id), csv_path))
            conn.commit()

    def get_classification_report(self):
        report_path = os.path.join(self.model_dir, "classification_report.txt")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                return f.read()
        return "Classification report not available."
