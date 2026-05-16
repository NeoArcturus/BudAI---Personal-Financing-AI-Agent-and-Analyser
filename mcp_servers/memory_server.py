import logging
from mcp.server.fastmcp import FastMCP
from services.memory_service import MemoryService
import json

logger = logging.getLogger("uvicorn.error")
mcp = FastMCP("Memory")

@mcp.tool()
def search_financial_history_semantic(query: str, user_uuid: str) -> str:
    mem = MemoryService()
    results = mem.semantic_search(query, user_uuid, limit=10)
    
    if not results or not results['documents']:
        return "No similar historical transactions found for this concept."
        
    formatted_results = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        formatted_results.append(f"- {doc} (Date: {meta['date']})")
        
    return "Found these relevant historical patterns:\n" + "\n".join(formatted_results)

if __name__ == "__main__":
    mcp.run()
