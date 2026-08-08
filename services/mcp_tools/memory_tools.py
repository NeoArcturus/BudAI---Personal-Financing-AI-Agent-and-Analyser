from services.memory_service import MemoryService
from services.logger_setup import get_core_logger

logger = get_core_logger("memory_tools")

def search_financial_history_semantic(query: str = "", user_uuid: str = "") -> str:
    """
    Perform a semantic search against the user's historical transactions and facts.
    
    Args:
        query (str, optional): The query to run against the memory store.
        user_uuid (str, optional): The unique identifier of the user.
        
    Returns:
        str: Relevant historical facts or past transaction patterns.
    """
    logger.info(f"Executing search_financial_history_semantic for user {user_uuid}")
    try:
        mem = MemoryService()
        results = mem.semantic_search(query, user_uuid, limit=10)
        
        if not results or not results['documents'] or not results['documents'][0]:
            _res = "No similar historical transactions found for this concept."
            logger.info(f"Tool returned: {str(_res)[:1000]}")
            return _res
            
        formatted_results = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            formatted_results.append(f"- {doc} (Date: {meta['date']})")
            
        _res = "Found these relevant historical patterns:\n" + "\n".join(formatted_results)
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res
    except Exception as e:
        logger.error(f"Error in search_financial_history_semantic: {e}")
        _res = f"Error searching history: {str(e)}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res

def get_seasonal_behavior_context(user_uuid: str = "") -> str:
    """
    Retrieve contextual parallels and spending behaviors for the current month based on past years.
    
    Args:
        user_uuid (str, optional): The unique identifier of the user.
        
    Returns:
        str: Contextual seasonal notes from the user's history.
    """
    logger.info(f"Executing get_seasonal_behavior_context for user {user_uuid}")
    try:
        mem = MemoryService()
        results = mem.get_seasonal_context(user_uuid, limit=10)
        
        if not results or not results['documents'] or not results['documents'][0]:
            _res = "No historical seasonal parallels found for this month."
            logger.info(f"Tool returned: {str(_res)[:1000]}")
            return _res
            
        context_docs = results['documents'][0]
        _res = " | ".join(context_docs)
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res
    except Exception as e:
        logger.error(f"Error in get_seasonal_behavior_context: {e}")
        _res = "Behavioral context currently unavailable."
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res

