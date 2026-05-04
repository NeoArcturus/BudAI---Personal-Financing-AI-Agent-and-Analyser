import sqlite3
import csv
import io
import os
import logging
from datetime import datetime
from langchain_core.tools import tool
from config import DATABASE_URL
from services.mcp_tools.tool_schema import (
    ExportAdvisoryStateInput, ExportAnalyzedStatementInput,
    MemorySearchInput, MemoryExtractionInput
)

logger = logging.getLogger("uvicorn.error")


@tool(args_schema=ExportAdvisoryStateInput)
def export_advisory_state(user_uuid: str, chart_type: str, raw_data: dict, ai_analysis: str) -> str:
    """Exports the latest operational data to the local filesystem for Advisory review."""
    try:
        from services.mcp_bridge import MCPBridge
        bridge = MCPBridge()
        path = bridge.write_advisory_file(
            user_uuid, chart_type, raw_data, ai_analysis)
        return f"Advisory state successfully secured at {path}. The UI can now load this file with zero hallucinations."
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=ExportAnalyzedStatementInput)
def export_custom_statement(user_uuid: str, ai_summary: str) -> str:
    """Generates a downloadable Excel/CSV statement populated with the user's categorized data and BudAI analysis."""
    try:
        db_path = DATABASE_URL.replace("sqlite:///", "")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT date, description, amount, category 
                FROM transactions 
                WHERE user_uuid = ? 
                ORDER BY date DESC
            """, (user_uuid,))
            rows = cursor.fetchall()

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

        bridge.generate_outbound_statement(file_path, csv_content)

        ui_tag = f"[TRIGGER_DOWNLOAD:{file_path}]"

        return f"Statement successfully generated at {file_path}. Use this exact trigger to prompt the UI download: {ui_tag}"

    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=MemorySearchInput)
def search_user_memory(query: str) -> str:
    """Searches the persistent Knowledge Graph for user preferences, past instructions, or entity context."""
    from services.mcp_bridge import MCPBridge
    bridge = MCPBridge()
    return bridge.interact_with_memory("search_nodes", {"query": query})


@tool(args_schema=MemoryExtractionInput)
def save_to_user_memory(entities: list) -> str:
    """Saves new facts, preferences, or observations into the persistent Knowledge Graph."""
    from services.mcp_bridge import MCPBridge
    bridge = MCPBridge()
    return bridge.interact_with_memory("create_entities", {"entities": entities})
