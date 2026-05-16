from fastapi import APIRouter, Depends, HTTPException
import time
import json
import re
from datetime import datetime, timedelta
from sqlalchemy import text
from typing import Any, Dict
from pydantic import BaseModel
from middleware.auth_middleware import get_current_user
from config import SessionLocal
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)


def _log_cache_hit(cache_id: str, chart_data: Any, query_type: str = "FRONTEND"):
    """Logs a summary of the cached data retrieval."""
    summary = {
        "keys": list(chart_data.keys()) if isinstance(chart_data, dict) else [],
        "labels_count": len(chart_data.get("labels", [])) if isinstance(chart_data, dict) and isinstance(chart_data.get("labels"), list) else "N/A",
        "datasets_count": len(chart_data.get("datasets", [])) if isinstance(chart_data, dict) and isinstance(chart_data.get("datasets"), list) else "N/A"
    }
    logger.warning(f" [CACHE_HIT:{query_type}] Retrieved {cache_id}. Data Summary: {summary}")

from services.mcp_tools.internal_tools import (
    generate_financial_forecast, classify_financial_data, find_total_spent_for_given_category,
    find_highest_spending_category, create_bargraph_chart_and_save, create_pie_chart_and_save,
    plot_expenses, generate_expense_forecast, analyze_critical_survival_metrics,
    analyze_wealth_acceleration_metrics, plot_cash_flow_mixed, plot_health_radar
)

media_router = APIRouter(prefix="/api/media", tags=["media"])


class MediaExecuteRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any] = {}


TOOL_MAPPING = {
    "generate_financial_forecast": generate_financial_forecast,
    "classify_financial_data": classify_financial_data,
    "find_total_spent_for_given_category": find_total_spent_for_given_category,
    "find_highest_spending_category": find_highest_spending_category,
    "create_bargraph_chart_and_save": create_bargraph_chart_and_save,
    "create_pie_chart_and_save": create_pie_chart_and_save,
    "plot_expenses": plot_expenses,
    "generate_expense_forecast": generate_expense_forecast,
    "analyze_critical_survival_metrics": analyze_critical_survival_metrics,
    "analyze_wealth_acceleration_metrics": analyze_wealth_acceleration_metrics,
    "plot_cash_flow_mixed": plot_cash_flow_mixed,
    "plot_health_radar": plot_health_radar
}


@media_router.post('/execute')
def execute_tool(request: MediaExecuteRequest, current_user=Depends(get_current_user)):
    start_time = time.time()
    tool_name = request.tool_name
    params = request.parameters
    user_uuid = current_user.user_uuid

    bank_name_or_id = params.get("bank_name_or_id", "")

    if isinstance(bank_name_or_id, str) and bank_name_or_id.startswith("CACHE_"):
        with SessionLocal() as session:
            row = session.execute(
                text("SELECT chart_data FROM chart_cache WHERE cache_id = :cache_id"),
                {"cache_id": bank_name_or_id}
            ).fetchone()
            if row:
                chart_data = json.loads(row[0])
                _log_cache_hit(bank_name_or_id, chart_data, query_type="FRONTEND")
                return {
                    "status": "success",
                    "metadata": {
                        "execution_time_ms": int((time.time() - start_time) * 1000),
                        "tool_executed": tool_name,
                        "query_type": "CACHED_RETRIEVAL",
                        "resolved_accounts": [{"bank_name": "Cached Data"}]
                    },
                    "data": chart_data
                }

    if tool_name not in TOOL_MAPPING:
        raise HTTPException(status_code=404, detail="Tool not found.")

    try:
        params["user_uuid"] = user_uuid
        tool_obj = TOOL_MAPPING[tool_name]

        if hasattr(tool_obj, "args"):
            if "from_date" in tool_obj.args and "from_date" not in params:
                params["from_date"] = (
                    datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            if "to_date" in tool_obj.args and "to_date" not in params:
                params["to_date"] = datetime.now().strftime("%Y-%m-%d")

        result_str = tool_obj.invoke(params)

        match = re.search(r'\[TRIGGER_[A-Z_]+:([^:\]]+)', str(result_str))
        if match:
            cache_id = match.group(1)
            with SessionLocal() as session:
                row = session.execute(
                    text("SELECT chart_data FROM chart_cache WHERE cache_id = :cache_id"),
                    {"cache_id": cache_id}
                ).fetchone()
                if row:
                    chart_data = json.loads(row[0])
                    _log_cache_hit(cache_id, chart_data, query_type="INTERNAL")
                    return {
                        "status": "success",
                        "metadata": {
                            "execution_time_ms": int((time.time() - start_time) * 1000),
                            "tool_executed": tool_name,
                            "query_type": "DYNAMIC_GENERATION",
                            "resolved_accounts": [{"bank_name": bank_name_or_id}]
                        },
                        "data": chart_data
                    }

        return {
            "status": "success",
            "metadata": {
                "execution_time_ms": int((time.time() - start_time) * 1000),
                "tool_executed": tool_name,
                "query_type": "TEXT_FALLBACK",
                "resolved_accounts": [{"bank_name": bank_name_or_id}]
            },
            "data": result_str
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
