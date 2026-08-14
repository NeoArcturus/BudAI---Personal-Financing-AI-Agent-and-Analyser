import json
import os
import faiss
import pickle
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
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
            
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(self.base_dir, "agent_cache", "faiss_memory")
        os.makedirs(self.db_path, exist_ok=True)
        
        self.index_file = os.path.join(self.db_path, "transactions.index")
        self.metadata_file = os.path.join(self.db_path, "metadata.pkl")
        
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
            
        logger.debug(json.dumps({"message": f"Initializing HuggingFaceEmbeddings locally", "status_code": 100}))
        self._model = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        
        dummy_emb = self._model.embed_query("init")
        self.embedding_dim = len(dummy_emb)
        logger.debug(json.dumps({"message": f"Dynamically determined embedding dimension: {self.embedding_dim}", "status_code": 100}))
        
        if os.path.exists(self.index_file):
            logger.debug(json.dumps({"message": f"Loading existing FAISS index from {self.index_file}", "status_code": 100}))
            self.index = faiss.read_index(self.index_file)
            if self.index.d != self.embedding_dim:
                logger.warning(json.dumps({"message": f"Existing FAISS index dimension ({self.index.d}) mismatch with model ({self.embedding_dim}). Creating new index.", "status_code": 400}))
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.metadata = []
        else:
            logger.debug(json.dumps({"message": f"Creating new FAISS IndexFlatL2", "status_code": 100}))
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            
        if os.path.exists(self.metadata_file) and (getattr(self.index, 'ntotal', 0) > 0 or not os.path.exists(self.index_file)):
            logger.debug(json.dumps({"message": f"Loading metadata from {self.metadata_file}", "status_code": 100}))
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
        elif not hasattr(self, 'metadata'):
            logger.debug(json.dumps({"message": f"Creating new metadata store", "status_code": 100}))
            self.metadata = []

        self._initialized = True
        logger.debug(json.dumps({"message": f"MemoryService initialization complete", "status_code": 100}))

    def _save(self):
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logger.error(json.dumps({"message": f"Failed to save MemoryService state: {e}", "status_code": 500}))

    def index_transactions(self, transactions, user_uuid):
        logger.debug(json.dumps({"message": f"Indexing {len(transactions) if transactions else 0} transactions for user {user_uuid}", "status_code": 100}))
        if not transactions:
            logger.debug(json.dumps({"message": f"No transactions to index", "status_code": 100}))
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
                embs_list = self._model.embed_documents(new_docs)
                embs = np.array(embs_list).astype('float32')
                self.index.add(embs)
                self.metadata.extend(new_metas)
                self._save()
                logger.debug(json.dumps({"message": f"Successfully indexed {len(new_docs)} transactions", "status_code": 100}))
            except Exception as e:
                logger.error(json.dumps({"message": f"FAISS indexing failed: {e}", "status_code": 500}))

    def semantic_search(self, query, user_uuid, limit=10):
        logger.debug(json.dumps({"message": f"Performing semantic search for user {user_uuid}", "status_code": 100}))
        try:
            query_emb_list = self._model.embed_query(query)
            query_emb = np.array([query_emb_list]).astype('float32')
            distances, indices = self.index.search(query_emb, limit * 5)
            
            results = {"documents": [[]], "metadatas": [[]]}
            count = 0
            for idx in indices[0]:
                if idx == -1: continue
                if idx >= len(self.metadata):
                    logger.warning(json.dumps({"message": f"Index {idx} out of metadata range", "status_code": 400}))
                    continue
                    
                meta = self.metadata[idx]
                if meta['user_uuid'] == user_uuid:
                    results["documents"][0].append(meta['text'])
                    results["metadatas"][0].append(meta)
                    count += 1
                if count >= limit:
                    break
            
            logger.debug(json.dumps({"message": f"Search complete. Found {count} relevant results", "status_code": 100}))
            return results
        except Exception as e:
            logger.error(json.dumps({"message": f"FAISS semantic search failed: {e}", "status_code": 500}))
            return {"documents": [[]], "metadatas": [[]]}

    def get_seasonal_context(self, user_uuid, limit=5):
        logger.debug(json.dumps({"message": f"Getting seasonal context for user {user_uuid}", "status_code": 100}))
        from datetime import datetime
        now = datetime.now()
        month_name = now.strftime('%B')
        
        query = f"Spending patterns in {month_name}"
        logger.debug(json.dumps({"message": f"Seasonal query generated: {query}", "status_code": 100}))
        return self.semantic_search(query, user_uuid, limit=limit)
