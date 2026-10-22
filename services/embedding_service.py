import os
from typing import List, Union
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                logger.info(f"Initializing FastEmbed model: {self.model_name}")
                self._model = TextEmbedding(model_name=self.model_name)
            except Exception as e:
                logger.error(f"Failed to load FastEmbed model {self.model_name}: {e}")
                raise
        return self._model

    def embed_query(self, text: str) -> List[float]:
        """
        Generates a 768-dimensional float embedding for a single text query.
        """
        if not text or not text.strip():
            return [0.0] * 768
        try:
            embeddings = list(self.model.embed([text]))
            return embeddings[0].tolist()
        except Exception as e:
            logger.error(f"Error generating embedding for query: {e}")
            return [0.0] * 768

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates 768-dimensional float embeddings for a batch of text documents.
        """
        if not texts:
            return []
        try:
            cleaned = [t if (t and t.strip()) else "empty" for t in texts]
            embeddings = list(self.model.embed(cleaned))
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings for batch: {e}")
            return [[0.0] * 768 for _ in texts]

_embedding_instance = None

def get_embedding_service() -> EmbeddingService:
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = EmbeddingService()
    return _embedding_instance
