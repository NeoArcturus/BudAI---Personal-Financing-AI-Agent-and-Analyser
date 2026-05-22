import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import uuid
from services.logger_setup import get_core_logger

logger = get_core_logger("memory_service")

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
        logger.debug(f"Using device: {self.device}")
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(self.base_dir, "agent_cache", "faiss_memory")
        os.makedirs(self.db_path, exist_ok=True)
        
        self.index_file = os.path.join(self.db_path, "transactions.index")
        self.metadata_file = os.path.join(self.db_path, "metadata.pkl")
        
        self.embedding_dim = 384
        
        if os.path.exists(self.index_file):
            logger.debug(f"Loading existing FAISS index from {self.index_file}")
            self.index = faiss.read_index(self.index_file)
        else:
            logger.info("Creating new FAISS IndexFlatL2")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            
        if os.path.exists(self.metadata_file):
            logger.debug(f"Loading metadata from {self.metadata_file}")
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            logger.info("Creating new metadata store")
            self.metadata = []

        logger.info("Loading SentenceTransformer model")
        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device)
        self._initialized = True
        logger.info("MemoryService initialization complete")

    def _save(self):
        logger.debug("Saving FAISS index and metadata")
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.debug("Save complete")
        except Exception as e:
            logger.error(f"Failed to save MemoryService state: {e}")

    def index_transactions(self, transactions, user_uuid):
        logger.info(f"Indexing {len(transactions) if transactions else 0} transactions for user {user_uuid}")
        if not transactions:
            logger.debug("No transactions to index")
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
            logger.debug(f"Encoding {len(new_docs)} documents")
            try:
                embs = self._model.encode(new_docs, convert_to_numpy=True).astype('float32')
                logger.debug("Adding embeddings to FAISS index")
                self.index.add(embs)
                self.metadata.extend(new_metas)
                self._save()
                logger.info(f"Successfully indexed {len(new_docs)} transactions")
            except Exception as e:
                logger.error(f"FAISS indexing failed: {e}")

    def semantic_search(self, query, user_uuid, limit=10):
        logger.info(f"Performing semantic search for user {user_uuid}")
        logger.debug(f"Query: {query}, Limit: {limit}")
        try:
            query_emb = self._model.encode([query], convert_to_numpy=True).astype('float32')
            logger.debug("Searching FAISS index")
            distances, indices = self.index.search(query_emb, limit * 5)
            
            results = {"documents": [[]], "metadatas": [[]]}
            count = 0
            logger.debug(f"Filtering {len(indices[0])} raw results")
            for idx in indices[0]:
                if idx == -1: continue
                if idx >= len(self.metadata):
                    logger.warning(f"Index {idx} out of metadata range")
                    continue
                    
                meta = self.metadata[idx]
                if meta['user_uuid'] == user_uuid:
                    results["documents"][0].append(meta['text'])
                    results["metadatas"][0].append(meta)
                    count += 1
                if count >= limit:
                    break
            
            logger.info(f"Search complete. Found {count} relevant results")
            return results
        except Exception as e:
            logger.error(f"FAISS semantic search failed: {e}")
            return {"documents": [[]], "metadatas": [[]]}

    def get_seasonal_context(self, user_uuid, limit=5):
        logger.info(f"Getting seasonal context for user {user_uuid}")
        import pandas as pd
        now = pd.Timestamp.utcnow()
        month_name = now.strftime('%B')
        
        query = f"Spending patterns in {month_name}"
        logger.debug(f"Seasonal query generated: {query}")
        return self.semantic_search(query, user_uuid, limit=limit)
