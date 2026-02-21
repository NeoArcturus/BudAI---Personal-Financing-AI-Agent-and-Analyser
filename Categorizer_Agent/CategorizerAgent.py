import os
import webbrowser
import pandas as pd
import sqlite3
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from api_integrator.access_token_generator import AccessTokenGenerator
from api_integrator.get_account_detail import UserAccount
from categorizer.categorizer import Categorizer
from categorizer.preprocessor import Preprocessor
from diskcache import Cache
from tqdm import tqdm
from dotenv import load_dotenv


class CategorizerAgent:
    def __init__(self, model_dir="saved_model", db_path="budai_memory.db"):
        self.model_dir = model_dir
        self.db_path = db_path
        self.local_st_path = os.path.join(model_dir, "st_model_local")
        self.cache = Cache('./agent_cache')
        self._init_db()
        self.token_gen = AccessTokenGenerator()
        self.categorizer = Categorizer()
        self._st_model = None

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    date TEXT,
                    amount REAL,
                    description TEXT,
                    category TEXT,
                    embedding BLOB
                )
            """)
            conn.commit()

    @property
    def st_model(self):
        if self._st_model is None:
            if not os.path.exists(self.local_st_path):
                os.makedirs(self.local_st_path, exist_ok=True)
                SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2").save(self.local_st_path)
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu")
            self._st_model = SentenceTransformer(
                self.local_st_path, device=device)
        return self._st_model

    def _authenticate(self):
        if not self.token_gen.regenerate_auth_token_using_refresh_token():
            webbrowser.open(self.token_gen.get_auth_link())
            self.token_gen.app.run(port=8080)

    def execute_cycle(self, start_date, end_date):
        self._authenticate()
        load_dotenv(override=True)
        self.user_acc = UserAccount()

        raw_df = self.user_acc.all_transactions(start_date, end_date)
        if raw_df is None or raw_df.empty:
            return None

        proc = Preprocessor(raw_df)
        clean_df = proc.preprocess()

        final_df = self.categorizer.categorize_data(clean_df)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "categorized_data.csv")

        print(f"Saving categorized data to: {csv_path}")
        final_df.to_csv(csv_path, index=False)

        descriptions = final_df['Description'].tolist()
        embeddings = self.st_model.encode(
            descriptions, batch_size=32, convert_to_numpy=True)

        self._update_sql_memory(final_df, embeddings)
        return final_df

    def _update_sql_memory(self, df, embs):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for i, row in df.iterrows():
                tx_id = f"tx_{i}_{row['Date']}"
                cursor.execute("""
                    INSERT OR REPLACE INTO transactions (id, date, amount, description, category, embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (tx_id, str(row['Date']), float(row['Amount']), row['Description'], row['Category'], embs[i].tobytes()))
            conn.commit()


if __name__ == "__main__":
    agent = CategorizerAgent()
    agent.execute_cycle("2025-01-01", "2026-01-01")
