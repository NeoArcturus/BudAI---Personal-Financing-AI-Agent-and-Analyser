import os
import webbrowser
import pandas as pd
import chromadb
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from api_integrator.access_token_generator import AccessTokenGenerator
from api_integrator.get_account_detail import UserAccount
from categorizer.categorizer import Categorizer
from categorizer.preprocessor import Preprocessor
from diskcache import Cache
from tqdm import tqdm


class FinancialAgent:
    def __init__(self, model_dir="saved_model"):
        self.model_dir = model_dir
        self.local_st_path = os.path.join(model_dir, "st_model_local")
        self.cache = Cache('../agent_cache')
        self.vector_db = chromadb.PersistentClient(path="../agent_vector_db")
        self.memory = self.vector_db.get_or_create_collection(
            name="fin_memory")
        self.token_gen = AccessTokenGenerator()
        self.user_acc = UserAccount()
        self.categorizer = Categorizer()
        self._st_model = None

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
        print("Checking authentication...")
        if not self.token_gen.regenerate_auth_token_using_refresh_token():
            webbrowser.open(self.token_gen.get_auth_link())
            self.token_gen.app.run(port=8080)

    def execute_cycle(self, start_date, end_date):
        print("Starting financial agent cycle...")
        print("Authenticating with Truelayer...")
        self._authenticate()

        raw_df = self.user_acc.all_transactions(start_date, end_date)
        if raw_df is None or raw_df.empty:
            print("No transactions found.")
            return None

        print("Preprocessing transaction data...")
        proc = Preprocessor(raw_df)
        clean_df = proc.preprocess()

        print("Generating embeddings and categorizing transactions...")
        final_df = self.categorizer.categorize_data(clean_df)
        final_df.to_csv("final_categorized_transactions.csv", index=False)

        print("Training categorizer model and categorizing data...")
        print("Loading categorizer model and categorizing transactions...")

        print("Generating embeddings for future forecasting...")
        descriptions = final_df['Description'].tolist()
        embeddings = self.st_model.encode(
            descriptions, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

        print("Updating vector memory...")
        self._update_vector_memory(final_df, embeddings)

        return final_df

    def _update_vector_memory(self, df, embs):
        texts = df['Description'].tolist()
        metas_df = df.copy()
        if 'Date' not in metas_df.columns:
            metas_df.loc[:, 'Date'] = pd.to_datetime(
                'today').strftime('%Y-%m-%d')

        cols_to_keep = [c for c in [
            'Amount', 'Category', 'Date'] if c in metas_df.columns]
        metas = metas_df[cols_to_keep].to_dict('records')
        ids = [f"tx_{i}_{d}" for i, d in enumerate(metas_df['Date'])]

        self.memory.upsert(
            ids=ids,
            embeddings=embs.tolist(),
            metadatas=metas,
            documents=texts
        )

    def get_memory(self, query, n_results=5):
        query_emb = self.st_model.encode([query]).tolist()
        return self.memory.query(query_embeddings=query_emb, n_results=n_results)


if __name__ == "__main__":
    agent = FinancialAgent()
    agent.execute_cycle("2022-10-13", "2026-02-01")
