import os
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from typing import Optional, List, Dict, Any
from config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_ENDPOINT_URL_S3,
    AWS_REGION,
    NEON_STORAGE_BUCKET
)
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class NeonStorageService:
    def __init__(self, bucket_name: Optional[str] = None):
        self.bucket = bucket_name or NEON_STORAGE_BUCKET or "assets"
        self._s3 = None
        if self.is_configured():
            try:
                self._s3 = boto3.client(
                    "s3",
                    endpoint_url=AWS_ENDPOINT_URL_S3,
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    region_name=AWS_REGION or "us-east-1",
                    config=Config(s3={"addressing_style": "path"})
                )
                logger.debug(f"Neon Storage Service initialized for bucket '{self.bucket}' at {AWS_ENDPOINT_URL_S3}")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}")
                self._s3 = None

    def is_configured(self) -> bool:
        return bool(AWS_ENDPOINT_URL_S3 and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)

    def upload_file(self, key: str, data: bytes | str, content_type: str = "application/octet-stream") -> Dict[str, Any]:
        if not self._s3:
            raise RuntimeError("Neon Object Storage is not configured.")
        
        body = data.encode("utf-8") if isinstance(data, str) else data
        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType=content_type
        )
        url = self.get_presigned_url(key)
        return {
            "bucket": self.bucket,
            "key": key,
            "size": len(body),
            "content_type": content_type,
            "presigned_url": url
        }

    def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        if not self._s3:
            raise RuntimeError("Neon Object Storage is not configured.")
        return self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in
        )

    def download_file(self, key: str) -> bytes:
        if not self._s3:
            raise RuntimeError("Neon Object Storage is not configured.")
        response = self._s3.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def list_objects(self, prefix: str = "") -> List[Dict[str, Any]]:
        if not self._s3:
            return []
        try:
            res = self._s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            contents = res.get("Contents", [])
            return [
                {
                    "key": item["Key"],
                    "size": item["Size"],
                    "last_modified": item["LastModified"].isoformat()
                }
                for item in contents
            ]
        except ClientError as e:
            logger.error(f"Error listing objects in bucket {self.bucket}: {e}")
            return []

    def delete_file(self, key: str) -> bool:
        if not self._s3:
            return False
        try:
            self._s3.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as e:
            logger.error(f"Error deleting object {key}: {e}")
            return False

# Global singleton instance
storage_service = NeonStorageService()
