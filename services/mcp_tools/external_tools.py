import csv
import io
import os
import logging
from datetime import datetime
from langchain_core.tools import tool
from config import SessionLocal
from models.database_models import Transaction
from services.mcp_tools.tool_schema import (
    ExportAdvisoryStateInput, ExportAnalyzedStatementInput,
    MemorySearchInput, MemoryExtractionInput
)
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)


@tool(args_schema=ExportAdvisoryStateInput)
def export_advisory_state(user_uuid: str, chart_type: str, raw_data: dict, ai_analysis: str) -> str:
    """Exports the latest operational data to the local filesystem for Advisory review."""
    logger.info(f"Exporting advisory state for user: {user_uuid}, chart: {chart_type}")
    try:
        from services.mcp_bridge import MCPBridge
        bridge = MCPBridge()
        path = bridge.write_advisory_file(
            user_uuid, chart_type, raw_data, ai_analysis)
        logger.info(f"Advisory state exported successfully to: {path}")
        return f"Advisory state successfully secured at {path}. The UI can now load this file with zero hallucinations."
    except Exception as e:
        logger.error(f"Error exporting advisory state: {str(e)}", exc_info=True)
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=ExportAnalyzedStatementInput)
def export_custom_statement(user_uuid: str, ai_summary: str) -> str:
    """Generates a downloadable Excel/CSV statement populated with the user's categorized data and BudAI analysis."""
    logger.info(f"Generating custom statement for user: {user_uuid}")
    try:
        with SessionLocal() as session:
            rows = session.query(
                Transaction.date, 
                Transaction.description, 
                Transaction.amount, 
                Transaction.category
            ).filter(Transaction.user_uuid == user_uuid).order_by(Transaction.date.desc()).all()

        logger.info(f"Fetched {len(rows)} transactions for user: {user_uuid}")
        payload = [{"Date": "AI ANALYSIS", "Description": ai_summary,
                    "Amount": "", "Category": "SYSTEM GENERATED"}]

        for row in rows:
            payload.append({
                "Date": str(row[0]),
                "Description": str(row[1]),
                "Amount": str(row[2]),
                "Category": str(row[3])
            })

        output = io.StringIO()
        writer = csv.DictWriter(
            output, fieldnames=["Date", "Description", "Amount", "Category"])
        writer.writeheader()
        writer.writerows(payload)
        csv_content = output.getvalue()

        from services.mcp_bridge import MCPBridge
        bridge = MCPBridge()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(bridge.workspace_dir, "exports")
        os.makedirs(export_dir, exist_ok=True)

        file_path = os.path.join(
            export_dir, f"BudAI_Analysis_{user_uuid}_{timestamp}.csv")

        logger.info(f"Writing CSV content to: {file_path}")
        bridge.generate_outbound_statement(file_path, csv_content)

        ui_tag = f"[TRIGGER_DOWNLOAD:{file_path}]"

        logger.info(f"Custom statement generated successfully for user: {user_uuid}")
        return f"Statement successfully generated at {file_path}. Use this exact trigger to prompt the UI download: {ui_tag}"

    except Exception as e:
        logger.error(f"Error generating custom statement: {str(e)}", exc_info=True)
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=MemorySearchInput)
def search_user_memory(query: str) -> str:
    """Searches the persistent Knowledge Graph for user preferences, past instructions, or entity context."""
    logger.info(f"Searching user memory with query: {query}")
    from services.mcp_bridge import MCPBridge
    bridge = MCPBridge()
    result = bridge.interact_with_memory("search_nodes", {"query": query})
    logger.debug(f"Memory search result: {result}")
    return result


@tool(args_schema=MemoryExtractionInput)
def save_to_user_memory(entities: list) -> str:
    """Saves new facts, preferences, or observations into the persistent Knowledge Graph."""
    logger.info(f"Saving {len(entities)} entities to user memory")
    from services.mcp_bridge import MCPBridge
    bridge = MCPBridge()
    result = bridge.interact_with_memory("create_entities", {"entities": entities})
    logger.info(f"Memory extraction result: {result}")
    return result
