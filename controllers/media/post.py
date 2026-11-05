from fastapi import HTTPException
import time
import json
import re
from datetime import datetime, timedelta
from sqlalchemy import text
from typing import Dict, Any, Optional
from pydantic import BaseModel
from models.database_models import User
from config import SessionLocal
from services.logger_setup import get_core_logger

from services.analytics.forecast_service import (
    compute_balance_forecast,
    compute_expense_forecast
)
from services.mcp_tools.forecaster_tools import generate_financial_forecast, generate_expense_forecast
from services.mcp_tools.core_query_tools import (
    query_transactions, aggregate_financial_data
)
from services.mcp_tools.health_tools import (
    analyze_critical_survival_metrics, analyze_wealth_acceleration_metrics,
    plot_health_radar, get_financial_health_metrics, _calculate_health_data
)

logger = get_core_logger(__name__)

class MediaExecuteRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any] = {}

TOOL_MAPPING = {
    "generate_financial_forecast": generate_financial_forecast,
    "generate_expense_forecast": generate_expense_forecast,
    "analyze_critical_survival_metrics": analyze_critical_survival_metrics,
    "analyze_wealth_acceleration_metrics": analyze_wealth_acceleration_metrics,
    "plot_health_radar": plot_health_radar,
    "get_financial_health_metrics": get_financial_health_metrics,
    "query_transactions": query_transactions,
    "aggregate_financial_data": aggregate_financial_data
}

async def execute_tool(request: MediaExecuteRequest, current_user: User):
    """
    Legacy dispatcher shim for /api/media/execute.
    Routes core operations directly to modern analytical service layers while
    maintaining backward compatibility for legacy clients.
    """
    start_time = time.time()
    tool_name = request.tool_name
    params = dict(request.parameters)
    user_uuid = current_user.user_uuid
    
    bank_name_or_id = params.get("bank_name_or_id")
    target_account = bank_name_or_id or params.get("account_id") or (params.get("account_ids") or [""])[0]
    
    # 1. Fast-path: Direct dispatch for balance forecast
    if tool_name == "generate_financial_forecast":
        try:
            days = int(params.get("days", 30))
            discipline = float(params.get("discipline_multiplier", 1.0))
            drift = float(params.get("drift_adjustment", 0.0))
            stress = bool(params.get("stress_test_active", False))
            macro = str(params.get("macro_environment", "Stable"))
            
            data = compute_balance_forecast(
                user_uuid=user_uuid,
                account_id=str(target_account),
                days=days,
                discipline_multiplier=discipline,
                drift_adjustment=drift,
                stress_test_active=stress,
                macro_environment=macro
            )
            return {
                "status": "success",
                "metadata": {
                    "execution_time_ms": int((time.time() - start_time) * 1000),
                    "tool_executed": tool_name,
                    "query_type": "DIRECT_SERVICE_EXECUTION",
                    "resolved_accounts": [{"bank_name": target_account or "Default"}]
                },
                "data": data
            }
        except Exception as e:
            logger.error(json.dumps({"message": f"Direct forecast execution failed: {e}", "status_code": 500}))
            raise HTTPException(status_code=500, detail=str(e))

    # 2. Fast-path: Direct dispatch for expense forecast
    if tool_name == "generate_expense_forecast":
        try:
            days = int(params.get("days", 30))
            data = compute_expense_forecast(
                user_uuid=user_uuid,
                account_id=str(target_account),
                days=days
            )
            return {
                "status": "success",
                "metadata": {
                    "execution_time_ms": int((time.time() - start_time) * 1000),
                    "tool_executed": tool_name,
                    "query_type": "DIRECT_SERVICE_EXECUTION",
                    "resolved_accounts": [{"bank_name": target_account or "Default"}]
                },
                "data": data
            }
        except Exception as e:
            logger.error(json.dumps({"message": f"Direct expense forecast failed: {e}", "status_code": 500}))
            raise HTTPException(status_code=500, detail=str(e))

    # 3. Fast-path: Direct dispatch for health metrics
    if tool_name == "get_financial_health_metrics":
        try:
            overall, metrics, recommendations = _calculate_health_data(user_uuid)
            return {
                "status": "success",
                "metadata": {
                    "execution_time_ms": int((time.time() - start_time) * 1000),
                    "tool_executed": tool_name,
                    "query_type": "DIRECT_SERVICE_EXECUTION",
                    "resolved_accounts": [{"bank_name": "All Accounts"}]
                },
                "data": json.dumps({
                    "overall_score": round(overall, 1),
                    "metrics": metrics,
                    "recommendations": recommendations[:3]
                })
            }
        except Exception as e:
            logger.error(json.dumps({"message": f"Direct health metrics failed: {e}", "status_code": 500}))
            raise HTTPException(status_code=500, detail=str(e))

    # 4. Cached retrieval fallback
    if isinstance(bank_name_or_id, str) and bank_name_or_id.startswith("CACHE_"):
        with SessionLocal() as session:
            row = session.execute(
                text("SELECT chart_data FROM chart_cache WHERE cache_id = :cache_id"),
                {"cache_id": bank_name_or_id}
            ).fetchone()
            if row:
                chart_data = json.loads(row[0])
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
        if target_account:
            params["account_id"] = str(target_account)
            params["account_ids"] = [str(target_account)]
        else:
            params.setdefault("account_id", "")
            params.setdefault("account_ids", [])
        
        tool_obj = TOOL_MAPPING[tool_name]
        
        if "from_date" not in params:
            params["from_date"] = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if "to_date" not in params:
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
                    return {
                        "status": "success",
                        "metadata": {
                            "execution_time_ms": int((time.time() - start_time) * 1000),
                            "tool_executed": tool_name,
                            "query_type": "DYNAMIC_GENERATION",
                            "resolved_accounts": [{"bank_name": bank_name_or_id or "Multiple"}]
                        },
                        "data": chart_data
                    }
        
        return {
            "status": "success",
            "metadata": {
                "execution_time_ms": int((time.time() - start_time) * 1000),
                "tool_executed": tool_name,
                "query_type": "TEXT_FALLBACK",
                "resolved_accounts": [{"bank_name": bank_name_or_id or "Multiple"}]
            },
            "data": result_str
        }
    except Exception as e:
        logger.error(json.dumps({"message": f"Error executing tool {tool_name} for user {user_uuid}: {e}", "status_code": 500}))
        raise HTTPException(status_code=500, detail=str(e))
