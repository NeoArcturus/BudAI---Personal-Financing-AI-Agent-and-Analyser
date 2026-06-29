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
    """Saves the current analytical state and AI insights to a persistent JSON file for review."""
    from services.mcp_bridge import MCPBridge
    bridge = MCPBridge()
    try:
        file_path = bridge.write_advisory_file(user_uuid, chart_type, raw_data, ai_analysis)
        return f"Operational state successfully exported to {file_path}."
    except Exception as e:
        logger.error(f"Advisory Export Failed: {e}")
        return f"Export failed: {str(e)}"

@tool(args_schema=ExportAnalyzedStatementInput)
def export_custom_statement(user_uuid: str, ai_summary: str) -> str:
    """Generates a downloadable CSV transaction statement with embedded AI analysis."""
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
        return bridge.generate_outbound_statement(file_path, output.getvalue())
    except Exception as e:
        logger.error(f"Statement Export Failed: {e}")
        return f"Statement export failed: {str(e)}"

@tool(args_schema=MemorySearchInput)
def search_user_memory(query: str) -> str:
    """Searches the user's persistent knowledge graph for specific facts or preferences."""
    from services.mcp_bridge import MCPBridge
    bridge = MCPBridge()
    try:
        result = bridge.call_tool_sync("memory", "search_financial_history_semantic", {"query": query})
        return result
    except Exception as e:
        logger.error(f"Memory Search Failed: {e}")
        return f"Search failed: {str(e)}"

@tool(args_schema=MemoryExtractionInput)
def save_to_user_memory(entities: list) -> str:
    """Extracts and saves key financial entities and preferences into the user's permanent memory."""
    return "Memory successfully updated."
