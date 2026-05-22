import warnings
import sys
warnings.filterwarnings("ignore")
from mcp.server.fastmcp import FastMCP
from services.memory_service import MemoryService
from services.logger_setup import get_core_logger
import json

logger = get_core_logger("memory_server")
mcp = FastMCP("Memory")

@mcp.tool()
def search_financial_history_semantic(query: str, user_uuid: str) -> str:
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

@mcp.tool()
def get_seasonal_behavior_context(user_uuid: str) -> str:
    """Retrieves semantic context about the user's historical behavior for the current month."""
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

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--sse", action="store_true", help="Run with SSE transport")
    args = parser.parse_args()

    if args.sse:
        logger.info(f"Starting Memory MCP Server with SSE transport")
        mcp.run(transport="sse")
    else:
        logger.info("Starting Memory MCP Server")
        try:
            mcp.run()
            logger.info("Memory MCP Server running")
        except Exception as e:
            logger.error(f"Failed to run Memory MCP Server: {e}")
            raise
