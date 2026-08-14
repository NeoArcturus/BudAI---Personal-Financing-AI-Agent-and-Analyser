import json
from fastapi import APIRouter, HTTPException
from services.memory_service import MemoryService
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)
router = APIRouter(prefix="/api/v1/memory", tags=["Memory"])

@router.get("/dump")
def dump_memory():
    """
    Returns the raw FAISS metadata containing all embedded transactions.
    """
    logger.info(json.dumps({"message": f"Memory dump endpoint called.", "status_code": 200}))
    try:
        mem = MemoryService()
        metadata = getattr(mem, 'metadata', [])
        return {
            "total_vectors": len(metadata),
            "dimensions": getattr(mem, 'embedding_dim', None),
            "data": metadata
        }
    except Exception as e:
        logger.error(json.dumps({"message": f"Error dumping memory: {e}", "status_code": 500}))
        raise HTTPException(status_code=500, detail=str(e))
