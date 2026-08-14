import json
from langchain_core.tools import tool
from config import SessionLocal
from models.database_models import Account
from pydantic import BaseModel, Field
from services.logger_setup import get_core_logger
from sqlalchemy import text

logger = get_core_logger(__name__)


class GetConnectedAccountsInput(BaseModel):
    user_uuid: str = Field(..., description="The user UUID to query accounts for.")

@tool(args_schema=GetConnectedAccountsInput)
def get_connected_accounts(user_uuid: str) -> str:
    """
    Use this tool to retrieve a list of all connected accounts and their account IDs for the user.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        
    Returns:
        str: A formatted string of connected accounts and their balances.
    """
    logger.info(json.dumps({"message": f"Executing MCP Tool: get_connected_accounts", "status_code": 200}))
    try:
        with SessionLocal() as session:
            accounts = session.execute(text("""
                SELECT b.bank_name, a.account_balance, a.currency, a.account_id 
                FROM accounts a
                JOIN banks b ON a.bank_uuid = b.bank_uuid
                WHERE a.user_uuid = :user_uuid
            """), {"user_uuid": user_uuid}).fetchall()
            
            if not accounts:
                _res = "No connected accounts found."
                logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
                return _res
            
            result = "Connected Accounts:\n"
            for acc in accounts:
                # Basic currency symbol mapping, fallback to the raw currency code
                currency_code = acc[2]
                symbol = "£" if currency_code == "GBP" else "$" if currency_code == "USD" else "€" if currency_code == "EUR" else currency_code + " "
                result += f"- Bank: {acc[0]} | Account ID: {acc[3]} | Balance: {symbol}{acc[1]}\n"
            _res = f"{result}\n\nDATA SUMMARY:\n{result}"
            logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
            return _res
    except Exception as e:
        _res = f"Database error: {str(e)}"
        logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
        return _res
