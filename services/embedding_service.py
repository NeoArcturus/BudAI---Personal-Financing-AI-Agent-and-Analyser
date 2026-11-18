import os
import gc
import time
from typing import List, Union
from services.logger_setup import get_core_logger, log_pipeline_data

logger = get_core_logger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.dimension = 384
        self.threads = int(os.getenv("FASTEMBED_THREADS", str(min(8, os.cpu_count() or 4))))
        self.cache_dir = os.getenv("FASTEMBED_CACHE_PATH", "/app/model_cache")
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                logger.info(f"Initializing FastEmbed model: {self.model_name} (threads={self.threads}, cache_dir={self.cache_dir})")
                cache = self.cache_dir if (self.cache_dir and os.path.exists(self.cache_dir)) else None
                self._model = TextEmbedding(
                    model_name=self.model_name,
                    threads=self.threads,
                    cache_dir=cache
                )
            except Exception as e:
                logger.error(f"Failed to load FastEmbed model {self.model_name}: {e}")
                raise
        return self._model

    def embed_query(self, text: str) -> List[float]:
        """
        Generates a 384-dimensional float embedding for a single text query using parallel ONNX threads.
        """
        if not text or not text.strip():
            return [0.0] * self.dimension
        try:
            embeddings = list(self.model.embed([text], batch_size=1, parallel=None))
            return embeddings[0].tolist()
        except Exception as e:
            logger.error(f"Error generating embedding for query: {e}")
            return [0.0] * self.dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates 384-dimensional float embeddings for a batch of text documents.
        Uses parallel ONNX C++ runtime execution across available CPU threads with data telemetry logging.
        """
        if not texts:
            return []
        t0 = time.time()
        try:
            cleaned = [t if (t and t.strip()) else "empty" for t in texts]
            batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
            embeddings = list(self.model.embed(cleaned, batch_size=batch_size, parallel=None))
            results = [emb.tolist() for emb in embeddings]
            del embeddings
            gc.collect()
            elapsed_ms = int((time.time() - t0) * 1000)
            log_pipeline_data(
                logger=logger,
                pipeline="vectorisation",
                stage="embed_documents",
                data={
                    "count": len(texts),
                    "model": self.model_name,
                    "threads": self.threads,
                    "batch_size": batch_size,
                    "elapsed_ms": elapsed_ms,
                    "dimension": self.dimension,
                    "sample": [t[:40] for t in cleaned[:3]]
                },
                message=f"Vectorised {len(texts)} documents in {elapsed_ms}ms (threads={self.threads}, dim={self.dimension})"
            )
            return results
        except Exception as e:
            logger.error(f"Error generating embeddings for batch: {e}")
            return [[0.0] * self.dimension for _ in texts]

_embedding_instance = None

def get_embedding_service() -> EmbeddingService:
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = EmbeddingService()
    return _embedding_instance
