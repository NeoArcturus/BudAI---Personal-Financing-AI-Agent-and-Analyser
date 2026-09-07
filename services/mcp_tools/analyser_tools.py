import json
import logging
from config import SessionLocal
from sqlalchemy import text
import pandas as pd
from langchain_core.tools import tool
from services.mcp_tools.shared_utils import (
    GetBudgetVarianceInput,
    _cache_chart_data, _parse_accounts, _get_combined_categorized_data
)
from agents.core_financial.Analyser_Agent.expense_analysis import ExpenseAnalysis
from agents.core_financial.Analyser_Agent.budget_engine import BudgetEngine
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

def _get_symbol(acc_uuid_or_name, user_uuid):
    try:
        with SessionLocal() as session:
            if acc_uuid_or_name:
                row = session.execute(text("SELECT a.currency FROM accounts a JOIN banks b ON a.bank_uuid = b.bank_uuid WHERE a.user_uuid = :u AND (a.account_id = :acc OR b.bank_name ILIKE :acc)"), {"u": user_uuid, "acc": acc_uuid_or_name}).fetchone()
            else:
                row = session.execute(text("SELECT currency FROM accounts WHERE user_uuid = :u LIMIT 1"), {"u": user_uuid}).fetchone()
            if row and row[0]:
                curr = row[0]
                return "£" if curr == "GBP" else "$" if curr == "USD" else "€" if curr == "EUR" else curr + " "
    except:
        pass
    return "£"


@tool(args_schema=GetBudgetVarianceInput)
def get_budget_variance(user_uuid: str, category: str = None) -> str:
    """
    Calculate and return the spend velocity, projected spend, and variance for the user's budgets.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        category (str, optional): The specific budget category to analyze.
        
    Returns:
        str: A breakdown of budget variance, spending velocity, and projected end-of-month spend.
    """
    logger.info(json.dumps({"message": f"Executing MCP Tool: get_budget_variance", "status_code": 200}))
    try:
        engine = BudgetEngine(user_uuid)
        sym = _get_symbol(None, user_uuid)
        results = engine.get_variance_for_category(category)
        
        if not results:
            _res = "No active budgets found for this category."
            logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
            return _res
            
        lines = []
        for r in results:
            lines.append(f"Category: {r['category']}")
            lines.append(f"  Monthly Limit: £{r['monthly_limit']:.2f}")
            lines.append(f"  Spent So Far: £{r['spent_so_far']:.2f}")
            lines.append(f"  Spend Velocity: {r['velocity']}x")
            lines.append(f"  Projected Month-End Spend: £{r['projected_spend']:.2f}")
            lines.append(f"  Variance: £{r['variance']:.2f}")
            lines.append("")
        
        _res = "\n".join(lines)
        logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
        return _res
    except Exception as e:
        logger.error(json.dumps({"message": f"Error calculating budget variance: {e}", "status_code": 500}))
        _res = f"Error: {str(e)}"
        logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
        return _res


