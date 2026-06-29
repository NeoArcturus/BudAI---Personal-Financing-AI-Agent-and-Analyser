from services.memory_service import MemoryService
from services.logger_setup import get_core_logger

logger = get_core_logger("memory_tools")

def search_financial_history_semantic(query: str = "", user_uuid: str = "") -> str:
    logger.info(f"Executing search_financial_history_semantic for user {user_uuid}")
    try:
        mem = MemoryService()
        results = mem.semantic_search(query, user_uuid, limit=10)
        
        if not results or not results['documents'] or not results['documents'][0]:
            return "No similar historical transactions found for this concept."
            
        formatted_results = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            formatted_results.append(f"- {doc} (Date: {meta['date']})")
            
        return "Found these relevant historical patterns:\n" + "\n".join(formatted_results)
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
