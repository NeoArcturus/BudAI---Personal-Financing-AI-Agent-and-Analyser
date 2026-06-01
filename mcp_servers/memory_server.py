import warnings
import sys
import os
import asyncio
from iii import register_worker
import json
from services.memory_service import MemoryService
from services.logger_setup import get_core_logger

warnings.filterwarnings("ignore")
logger = get_core_logger("memory_server")

III_ENGINE_URL = os.getenv("III_ENGINE_URL", "ws://iii-engine:49134")

def search_financial_history_semantic(query: str = "", user_uuid: str = "") -> str:
    logger.info(f"Executing search_financial_history_semantic for user {user_uuid}")
    logger.debug(f"Query: {query}")
    try:
        mem = MemoryService()
        logger.debug("MemoryService initialized")
        results = mem.semantic_search(query, user_uuid, limit=10)
        
        if not results or not results['documents'] or not results['documents'][0]:
            logger.info("No semantic results found")
            return "No similar historical transactions found for this concept."
            
        formatted_results = []
        logger.debug(f"Found {len(results['documents'][0])} results")
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            formatted_results.append(f"- {doc} (Date: {meta['date']})")
            
        result_str = "Found these relevant historical patterns:\n" + "\n".join(formatted_results)
        logger.debug("Successfully formatted results")
        return result_str
    except Exception as e:
        logger.error(f"Error in search_financial_history_semantic: {e}")
        return f"Error searching history: {str(e)}"

def get_seasonal_behavior_context(user_uuid: str = "") -> str:
    logger.info(f"Executing get_seasonal_behavior_context for user {user_uuid}")
    try:
        mem = MemoryService()
        results = mem.get_seasonal_context(user_uuid, limit=10)
        
        if not results or not results['documents'] or not results['documents'][0]:
            return "No historical seasonal parallels found for this month."
            
        context_docs = results['documents'][0]
        return " | ".join(context_docs)
    except Exception as e:
        logger.error(f"Error in get_seasonal_behavior_context: {e}")
        return "Behavioral context currently unavailable."

def tool_wrapper(func, args):
    # Filter out internal iii arguments
    clean_args = {k: v for k, v in args.items() if not k.startswith("_")}
    try:
        if asyncio.iscoroutinefunction(func):
            try:
                return asyncio.run(func(**clean_args))
            except RuntimeError:
                # Fallback if loop is already running
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(func(**clean_args))
        else:
            return func(**clean_args)
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {e}", exc_info=True)
        return f"Error: {str(e)}"

async def main():
    logger.info("Initializing Memory Worker with iii")
    worker = register_worker(III_ENGINE_URL)
    
    worker.register_function("memory::search_financial_history_semantic", lambda args: tool_wrapper(search_financial_history_semantic, args))
    worker.register_function("memory::get_seasonal_behavior_context", lambda args: tool_wrapper(get_seasonal_behavior_context, args))
    
    logger.info("Memory Worker registered. Entering heartbeat loop.")
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
