import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import uuid
import logging

logger = logging.getLogger("uvicorn.error")

class MemoryService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemoryService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(self.base_dir, "agent_cache", "faiss_memory")
        os.makedirs(self.db_path, exist_ok=True)
        
        self.index_file = os.path.join(self.db_path, "transactions.index")
        self.metadata_file = os.path.join(self.db_path, "metadata.pkl")
        
        self.embedding_dim = 384 # all-MiniLM-L6-v2 dimension
        
        # Initialize FAISS Index
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            
        # Initialize Metadata Store
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = [] # List of dicts mapping index to tx info

        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device)
        self._initialized = True

    def _save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)

    def index_transactions(self, transactions, user_uuid):
        if not transactions:
            return
            
        new_docs = []
        new_metas = []
        
        for tx in transactions:
            tx_id = tx.get('transaction_uuid') or str(uuid.uuid4())
            desc = tx.get('description', '')
            category = tx.get('category', 'Uncategorized')
            amount = tx.get('amount', 0.0)
            date = tx.get('date').isoformat() if hasattr(tx.get('date'), 'isoformat') else str(tx.get('date'))
            
            doc_text = f"{desc} {category} £{abs(amount):.2f}"
            
            new_docs.append(doc_text)
            new_metas.append({
                "tx_id": tx_id,
                "user_uuid": user_uuid,
                "date": date,
                "amount": amount,
                "category": category,
                "text": doc_text
            })
            
        if new_docs:
            try:
                embs = self._model.encode(new_docs, convert_to_numpy=True).astype('float32')
                self.index.add(embs)
                self.metadata.extend(new_metas)
                self._save()
            except Exception as e:
                logger.error(f"FAISS indexing failed: {e}")

    def semantic_search(self, query, user_uuid, limit=10):
        try:
            query_emb = self._model.encode([query], convert_to_numpy=True).astype('float32')
            distances, indices = self.index.search(query_emb, limit * 5) # Search more to filter by user_uuid
            
            # Filter results by user_uuid
            results = {"documents": [[]], "metadatas": [[]]}
            count = 0
            for idx in indices[0]:
                if idx == -1: continue
                meta = self.metadata[idx]
                if meta['user_uuid'] == user_uuid:
                    results["documents"][0].append(meta['text'])
                    results["metadatas"][0].append(meta)
                    count += 1
                if count >= limit:
                    break
            return results
        except Exception as e:
            logger.error(f"FAISS semantic search failed: {e}")
            return {"documents": [[]], "metadatas": [[]]}

    def get_seasonal_context(self, user_uuid, limit=5):
        import pandas as pd
        now = pd.Timestamp.utcnow()
        month_name = now.strftime('%B')
        
        query = f"Spending patterns in {month_name}"
        return self.semantic_search(query, user_uuid, limit=limit)
