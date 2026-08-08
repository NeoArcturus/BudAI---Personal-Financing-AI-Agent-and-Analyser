import threading
from fastapi import HTTPException

class GlobalLLMManager:
    _lock = threading.Lock()

    @classmethod
    def try_acquire(cls) -> bool:
        """
        Attempts to acquire the lock for frontend requests.
        Raises 423 if the LLM is currently busy processing background tasks.
        """
        acquired = cls._lock.acquire(blocking=False)
        if not acquired:
            raise HTTPException(
                status_code=423, 
                detail="The AI Engine is currently processing background tasks. Please wait."
            )
        return True

    @classmethod
    def acquire_background(cls):
        """
        Used by background batch processes. Will block and wait for 
        any existing task to finish, ensuring sequential pipeline.
        """
        cls._lock.acquire(blocking=True)

    @classmethod
    def release(cls):
        cls._lock.release()
