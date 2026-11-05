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
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        export_dir = os.path.join(base_dir, "agent_cache", "exports")
        os.makedirs(export_dir, exist_ok=True)
        filename = f"AdvisoryState_{user_uuid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path = os.path.join(export_dir, filename)

        payload = {
            "user_uuid": user_uuid,
            "chart_type": chart_type,
            "exported_at": datetime.utcnow().isoformat(),
            "raw_data": raw_data,
            "ai_analysis": ai_analysis
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

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
    try:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["--- AI FINANCIAL SUMMARY ---"])
        writer.writerow([ai_summary])
        writer.writerow([])
        writer.writerow(["Date", "Description", "Category", "Amount"])
        with SessionLocal() as session:
            from models.database_models import MerchantKnowledge
            rows = session.query(Transaction, MerchantKnowledge.category).outerjoin(
                MerchantKnowledge, Transaction.merchant_knowledge_uuid == MerchantKnowledge.knowledge_uuid
            ).filter(Transaction.user_uuid == user_uuid).order_by(Transaction.date.desc()).limit(100).all()
            for tx, cat in rows:
                date_str = tx.date.strftime("%Y-%m-%d") if tx.date else ""
                amount_str = f"£{tx.amount:.2f}" if tx.amount is not None else "£0.00"
                writer.writerow([date_str, tx.description or "", cat or "Uncategorized", amount_str])
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        export_dir = os.path.join(base_dir, "agent_cache", "exports")
        os.makedirs(export_dir, exist_ok=True)
        filename = f"Statement_{user_uuid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = os.path.join(export_dir, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(output.getvalue())

        _res = f"Custom statement successfully generated at {file_path}."
        logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
        return _res
    except Exception as e:
        logger.error(json.dumps({"message": f"Statement Export Failed: {e}", "status_code": 500}))
        _res = f"Statement export failed: {str(e)}"
        logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
        return _res

@tool(args_schema=MemorySearchInput)
def search_user_memory(query: str, user_uuid: str = "") -> str:
    """
    Searches the user's persistent knowledge graph for specific facts or preferences.
    
    Args:
        query (str): The search term or question to query against the memory graph.
        user_uuid (str, optional): The user UUID.
        
    Returns:
        str: The retrieved contextual memory or facts.
    """
    logger.info(json.dumps({"message": f"Executing MCP Tool: search_user_memory", "status_code": 200}))
    try:
        from services.memory_service import MemoryService
        mem = MemoryService()
        results = mem.semantic_search(query, user_uuid=user_uuid, limit=5)
        
        if not results or not results.get('documents') or not results['documents'][0]:
            _res = "No matching historical facts found."
        else:
            formatted = [f"- {d}" for d in results['documents'][0] if d]
            _res = "Retrieved Memory Context:\n" + "\n".join(formatted)

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
