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
        
        base_url = os.getenv("LLM_BASE_URL", "http://host.docker.internal:8000/v1")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
            
        logger.debug("Initializing HuggingFaceEmbeddings locally")
        self._model = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        
        dummy_emb = self._model.embed_query("init")
        self.embedding_dim = len(dummy_emb)
        logger.debug(f"Dynamically determined embedding dimension: {self.embedding_dim}")
        
        if os.path.exists(self.index_file):
            logger.debug(f"Loading existing FAISS index from {self.index_file}")
            self.index = faiss.read_index(self.index_file)
            if self.index.d != self.embedding_dim:
                logger.warning(f"Existing FAISS index dimension ({self.index.d}) mismatch with model ({self.embedding_dim}). Creating new index.")
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.metadata = []
        else:
            logger.debug("Creating new FAISS IndexFlatL2")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            
        if os.path.exists(self.metadata_file) and (getattr(self.index, 'ntotal', 0) > 0 or not os.path.exists(self.index_file)):
            logger.debug(f"Loading metadata from {self.metadata_file}")
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
        elif not hasattr(self, 'metadata'):
            logger.debug("Creating new metadata store")
            self.metadata = []

        self._initialized = True
        logger.debug("MemoryService initialization complete")

    def _save(self):
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Failed to save MemoryService state: {e}")

    def index_transactions(self, transactions, user_uuid):
        logger.debug(f"Indexing {len(transactions) if transactions else 0} transactions for user {user_uuid}")
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
            try:
                embs_list = self._model.embed_documents(new_docs)
                embs = np.array(embs_list).astype('float32')
                self.index.add(embs)
                self.metadata.extend(new_metas)
                self._save()
                logger.debug(f"Successfully indexed {len(new_docs)} transactions")
            except Exception as e:
                logger.error(f"FAISS indexing failed: {e}")

    def semantic_search(self, query, user_uuid, limit=10):
        logger.debug(f"Performing semantic search for user {user_uuid}")
        try:
            query_emb_list = self._model.embed_query(query)
            query_emb = np.array([query_emb_list]).astype('float32')
            distances, indices = self.index.search(query_emb, limit * 5)
            
            results = {"documents": [[]], "metadatas": [[]]}
            count = 0
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
            
            logger.debug(f"Search complete. Found {count} relevant results")
            return results
        except Exception as e:
            logger.error(f"FAISS semantic search failed: {e}")
            return {"documents": [[]], "metadatas": [[]]}

    def get_seasonal_context(self, user_uuid, limit=5):
        logger.debug(f"Getting seasonal context for user {user_uuid}")
        from datetime import datetime
        now = datetime.now()
        month_name = now.strftime('%B')
        
        query = f"Spending patterns in {month_name}"
        logger.debug(f"Seasonal query generated: {query}")
        return self.semantic_search(query, user_uuid, limit=limit)
