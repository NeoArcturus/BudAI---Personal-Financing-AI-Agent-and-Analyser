import json
from fastapi import HTTPException
from config import redis_client

async def get_summarize_status(job_id: str):
    """
    Retrieves the status of an asynchronous summarize task from Redis.
    
    Args:
        job_id (str): The UUID of the background job.
        
    Returns:
        dict: The job status payload containing status and optional insight.
        
    Raises:
        HTTPException (404): If the job ID does not exist in Redis.
    """
    job_raw = redis_client.get(f"job:{job_id}")
    if not job_raw:
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(job_raw)
