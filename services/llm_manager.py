import os
import threading
from fastapi import HTTPException

class GlobalLLMManager:
    _max_concurrency = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))
    _semaphore = threading.Semaphore(_max_concurrency)

    @classmethod
    def try_acquire(cls) -> bool:
        """
        Attempts to acquire the lock for frontend requests.
        Raises 423 if the LLM is currently busy processing background tasks.
        """
        acquired = cls._semaphore.acquire(blocking=False)
        if not acquired:
            raise HTTPException(
                status_code=423, 
                detail="The AI Engine is currently operating at maximum capacity. Please wait."
            )
        return True

    @classmethod
    def acquire_background(cls):
        """
        Used by background batch processes. Will block and wait for 
        any existing task to finish, ensuring sequential pipeline.
        """
        cls._semaphore.acquire(blocking=True)

    @classmethod
    def release(cls):
        """
        Releases the acquired semaphore permit back to the internal counter.
        """
        cls._semaphore.release()