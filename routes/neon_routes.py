from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Optional
from sqlalchemy import text
from config import (
    NEON_BRANCH,
    NEON_AUTH_BASE_URL,
    NEON_AUTH_JWKS_URL,
    NEON_STORAGE_BUCKET,
    AWS_ENDPOINT_URL_S3,
    get_db,
    engine
)
from services.storage_service import storage_service
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

router = APIRouter(prefix="/api/neon", tags=["neon"])


@router.get("/status")
def get_neon_status(db=Depends(get_db)):
    """
    Returns live connectivity status of configured Neon primitives:
    Lakebase Postgres, Neon Auth, and Neon Object Storage.
    """
    db_ok = False
    db_version = None
    try:
        with engine.connect() as conn:
            res = conn.execute(text("SELECT version();")).scalar()
            db_ok = True
            db_version = res
    except Exception as e:
        logger.error(f"Postgres health check failed: {e}")

    storage_ok = storage_service.is_configured()
    bucket_count = 0
    if storage_ok:
        try:
            objs = storage_service.list_objects()
            bucket_count = len(objs)
        except Exception:
            storage_ok = False

    return {
        "branch": NEON_BRANCH,
        "services": {
            "postgres": {
                "configured": True,
                "connected": db_ok,
                "version_summary": db_version[:80] if db_version else None
            },
            "auth": {
                "configured": bool(NEON_AUTH_BASE_URL),
                "base_url": NEON_AUTH_BASE_URL,
                "jwks_url": NEON_AUTH_JWKS_URL
            },
            "object_storage": {
                "configured": storage_service.is_configured(),
                "endpoint": AWS_ENDPOINT_URL_S3,
                "bucket": NEON_STORAGE_BUCKET,
                "objects_cached": bucket_count
            }
        }
    }


@router.get("/storage/list")
def list_storage_objects(prefix: str = ""):
    """Lists objects in the Neon Object Storage bucket."""
    try:
        return {"bucket": storage_service.bucket, "objects": storage_service.list_objects(prefix)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/storage/upload")
async def upload_storage_file(
    file: UploadFile = File(...),
    path_prefix: Optional[str] = Form("uploads")
):
    """Uploads a file to Neon Object Storage and returns its metadata & presigned URL."""
    try:
        content = await file.read()
        key = f"{path_prefix}/{file.filename}" if path_prefix else file.filename
        res = storage_service.upload_file(key, content, content_type=file.content_type or "application/octet-stream")
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/storage/url/{key:path}")
def get_object_presigned_url(key: str):
    """Generates a secure presigned URL for an object in Neon Object Storage."""
    try:
        url = storage_service.get_presigned_url(key)
        return {"key": key, "presigned_url": url}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
