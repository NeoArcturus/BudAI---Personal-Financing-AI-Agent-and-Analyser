import os
from typing import Dict, Any, List, Optional, Union
from qdrant_client import QdrantClient, models
from config import QDRANT_URL, QDRANT_API_KEY
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class QdrantService:
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        self.url = url or QDRANT_URL
        self.api_key = api_key or QDRANT_API_KEY
        self._client: Optional[QdrantClient] = None

    def is_configured(self) -> bool:
        return bool(self.url and self.api_key)

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            if not self.is_configured():
                raise ValueError("Qdrant credentials (QDRANT_URL and QDRANT_API_KEY) are not configured.")
            self._client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=30.0
            )
        return self._client

    def check_connection(self) -> Dict[str, Any]:
        """
        Validates connection to the Qdrant Cloud cluster and returns metadata.
        """
        try:
            collections_response = self.client.get_collections()
            collection_names = [col.name for col in collections_response.collections]
            return {
                "status": "connected",
                "url": self.url,
                "collections_count": len(collection_names),
                "collections": collection_names
            }
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant cluster: {e}")
            return {
                "status": "error",
                "url": self.url,
                "error": str(e)
            }

    def ensure_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
        enable_quantization: bool = True
    ) -> bool:
        """
        Ensures a collection exists with optimized settings for high performance and low memory.
        """
        try:
            dist_map = {
                "Cosine": models.Distance.COSINE,
                "Euclid": models.Distance.EUCLID,
                "Dot": models.Distance.DOT
            }
            target_distance = dist_map.get(distance, models.Distance.COSINE)

            existing = [c.name for c in self.client.get_collections().collections]
            if collection_name in existing:
                logger.info(f"Collection {collection_name} already exists.")
                return True

            quantization_config = None
            if enable_quantization:
                quantization_config = models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True
                    )
                )

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=target_distance,
                    on_disk=True
                ),
                quantization_config=quantization_config
            )
            logger.info(f"Created Qdrant collection {collection_name} (size={vector_size}, distance={distance}).")
            return True
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            raise

    def upsert_vectors(
        self,
        collection_name: str,
        points: List[models.PointStruct]
    ) -> Dict[str, Any]:
        """
        Inserts or updates points in the specified collection.
        """
        try:
            res = self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            return {
                "status": "success",
                "operation_id": res.operation_id
            }
        except Exception as e:
            logger.error(f"Error upserting vectors into {collection_name}: {e}")
            raise

    def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        query_filter: Optional[models.Filter] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches for nearest vectors using HNSW index with optional filtering.
        """
        try:
            search_result = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit
            )
            return [
                {
                    "id": p.id,
                    "score": p.score,
                    "payload": p.payload
                }
                for p in search_result.points
            ]
        except Exception as e:
            logger.error(f"Error searching vectors in {collection_name}: {e}")
            raise

    def upsert_merchant(
        self,
        knowledge_uuid: str,
        clean_merchant_name: str,
        category: str,
        vector: List[float],
        sub_category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_human_verified: bool = False
    ) -> bool:
        """
        Upserts a merchant record into the Qdrant merchant_knowledge collection.
        """
        import uuid
        try:
            point_id = str(uuid.UUID(str(knowledge_uuid)))
        except ValueError:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(knowledge_uuid)))

        point = models.PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "knowledge_uuid": str(knowledge_uuid),
                "clean_merchant_name": clean_merchant_name,
                "category": category,
                "sub_category": sub_category or "",
                "tags": tags or [],
                "is_human_verified": is_human_verified
            }
        )
        self.upsert_vectors("merchant_knowledge", [point])
        return True

    def upsert_merchants_batch(
        self,
        merchants: List[Dict[str, Any]]
    ) -> bool:
        """
        Batch upserts multiple merchant records into Qdrant.
        """
        import uuid
        points = []
        for m in merchants:
            k_uuid = str(m["knowledge_uuid"])
            try:
                point_id = str(uuid.UUID(k_uuid))
            except ValueError:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, k_uuid))

            point = models.PointStruct(
                id=point_id,
                vector=m["vector"],
                payload={
                    "knowledge_uuid": k_uuid,
                    "clean_merchant_name": m["clean_merchant_name"],
                    "category": m.get("category", "Uncategorized"),
                    "sub_category": m.get("sub_category", "") or "",
                    "tags": m.get("tags", []) or [],
                    "is_human_verified": m.get("is_human_verified", False)
                }
            )
            points.append(point)

        if points:
            self.upsert_vectors("merchant_knowledge", points)
        return True

    def update_merchant_payload(
        self,
        knowledge_uuid: str,
        category: str,
        sub_category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_human_verified: bool = False
    ) -> bool:
        """
        Updates the category/payload of an existing merchant in Qdrant.
        """
        import uuid
        try:
            point_id = str(uuid.UUID(str(knowledge_uuid)))
        except ValueError:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(knowledge_uuid)))

        payload_update = {
            "category": category,
            "is_human_verified": is_human_verified
        }
        if sub_category is not None:
            payload_update["sub_category"] = sub_category
        if tags is not None:
            payload_update["tags"] = tags

        try:
            self.client.set_payload(
                collection_name="merchant_knowledge",
                payload=payload_update,
                points=[point_id]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update merchant payload in Qdrant for {knowledge_uuid}: {e}")
            return False

    def search_merchant_knowledge(
        self,
        query_vector: List[float],
        limit: int = 1,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches merchant_knowledge in Qdrant for semantic matches.
        """
        results = self.search_vectors("merchant_knowledge", query_vector=query_vector, limit=limit)
        if score_threshold is not None:
            results = [r for r in results if r.get("score", 0.0) >= score_threshold]
        return results

_service_instance: Optional[QdrantService] = None

def get_qdrant_service() -> QdrantService:
    global _service_instance
    if _service_instance is None:
        _service_instance = QdrantService()
    return _service_instance
