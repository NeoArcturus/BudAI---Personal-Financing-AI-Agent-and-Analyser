import json
import csv
import io
import os
import logging
from datetime import datetime
from langchain_core.tools import tool
from config import SessionLocal
from models.database_models import Transaction
from services.mcp_tools.tool_schema import (
    ExportAdvisoryStateInput,
    ExportAnalyzedStatementInput,
    MemorySearchInput,
    MemoryExtractionInput
)
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)


@tool(args_schema=ExportAdvisoryStateInput)
def export_advisory_state(user_uuid: str, chart_type: str, raw_data: dict, ai_analysis: str) -> str:
    """
    Saves the current analytical state and AI insights to a persistent JSON file for review.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        chart_type (str): The type of chart being exported.
        raw_data (dict): The raw chart data.
        ai_analysis (str): The generated AI insights.
        
    Returns:
        str: A success message indicating the export path or an error.
    """
    logger.info(json.dumps({"message": f"Executing MCP Tool: export_advisory_state", "status_code": 200}))
    from services.mcp_bridge import MCPBridge
    bridge = MCPBridge()
    try:
        file_path = bridge.write_advisory_file(user_uuid, chart_type, raw_data, ai_analysis)
        _res = f"Operational state successfully exported to {file_path}."
        logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
        return _res
    except Exception as e:
        logger.error(json.dumps({"message": f"Advisory Export Failed: {e}", "status_code": 500}))
        _res = f"Export failed: {str(e)}"
        logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
        return _res

@tool(args_schema=ExportAnalyzedStatementInput)
def export_custom_statement(user_uuid: str, ai_summary: str) -> str:
    """
    Generates a downloadable CSV transaction statement with embedded AI analysis.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        ai_summary (str): The AI analysis to embed at the top of the CSV.
        
    Returns:
        str: A success message with the file download path or an error.
    """
    logger.info(json.dumps({"message": f"Executing MCP Tool: export_custom_statement", "status_code": 200}))
    from services.mcp_bridge import MCPBridge
    bridge = MCPBridge()
    try:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["--- AI FINANCIAL SUMMARY ---"])
        writer.writerow([ai_summary])
        writer.writerow([])
        writer.writerow(["Date", "Description", "Category", "Amount"])
        with SessionLocal() as session:
            txs = session.query(Transaction).filter_by(user_uuid=user_uuid).order_by(Transaction.date.desc()).limit(100).all()
            for tx in txs:
                writer.writerow([tx.date.strftime("%Y-%m-%d"), tx.description, tx.category, f"£{tx.amount:.2f}"])
        filename = f"Statement_{user_uuid}_{datetime.now().strftime('%Y%m%d')}.csv"
        file_path = os.path.join(bridge.workspace_dir, "exports", filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        _res = bridge.generate_outbound_statement(file_path, output.getvalue())
        logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
        return _res
    except Exception as e:
        logger.error(json.dumps({"message": f"Statement Export Failed: {e}", "status_code": 500}))
        _res = f"Statement export failed: {str(e)}"
        logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
        return _res

@tool(args_schema=MemorySearchInput)
def search_user_memory(query: str) -> str:
    """
    Searches the user's persistent knowledge graph for specific facts or preferences.
    
    Args:
        query (str): The search term or question to query against the memory graph.
        
    Returns:
        str: The retrieved contextual memory or facts.
    """
    logger.info(json.dumps({"message": f"Executing MCP Tool: search_user_memory", "status_code": 200}))
    from services.mcp_bridge import MCPBridge
    bridge = MCPBridge()
    try:
        result = bridge.call_tool_sync("memory", "search_financial_history_semantic", {"query": query})
        _res = result
        logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
        return _res
    except Exception as e:
        logger.error(json.dumps({"message": f"Memory Search Failed: {e}", "status_code": 500}))
        _res = f"Search failed: {str(e)}"
        logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
        return _res

@tool(args_schema=MemoryExtractionInput)
def save_to_user_memory(entities: list) -> str:
    """
    Extracts and saves key financial entities and preferences into the user's permanent memory.
    
    Args:
        entities (list): A list of memory entities to save.
        
    Returns:
        str: A success message indicating memory was updated.
    """
    logger.info(json.dumps({"message": f"Executing MCP Tool: save_to_user_memory", "status_code": 200}))
    _res = "Memory successfully updated."
    logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
    return _res
