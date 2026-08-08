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
    logger.info("Memory dump endpoint called.")
    try:
        mem = MemoryService()
        metadata = getattr(mem, 'metadata', [])
        return {
            "total_vectors": len(metadata),
            "dimensions": getattr(mem, 'embedding_dim', None),
            "data": metadata
        }
    except Exception as e:
        logger.error(f"Error dumping memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))
