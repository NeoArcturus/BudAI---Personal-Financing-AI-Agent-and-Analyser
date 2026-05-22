import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "budai_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300, # 5 minutes max for AI processing
)

# Auto-discover tasks from the workers or services
@celery_app.task(name="execute_langgraph_workflow", bind=True)
def execute_langgraph_workflow(self, state_input):
    """
    Background task to execute the complex LangGraph orchestrator.
    """
    import asyncio
    from services.orchestrator_graph import execute_chat_graph
    
    # Setup async loop for the worker thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # We need a non-streaming version or a way to collect results for the queue
        # For the queue version, we collect all tokens and return final result
        q = execute_chat_graph(state_input)
        full_response = []
        
        while True:
            token = q.get()
            if token is None:
                break
            full_response.append(token)
            
        return "".join(full_response)
    finally:
        loop.close()
