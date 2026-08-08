from langchain_core.tools import tool
from config import SessionLocal
from models.database_models import Account
from pydantic import BaseModel, Field
from services.logger_setup import get_core_logger

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
    logger.info(f"Executing MCP Tool: get_connected_accounts")
    try:
        with SessionLocal() as session:
            accounts = session.query(Account).filter_by(user_uuid=user_uuid).all()
            if not accounts:
                _res = "No connected accounts found."
                logger.info(f"Tool returned: {str(_res)[:1000]}")
                return _res
            
            result = "Connected Accounts:\n"
            for acc in accounts:
                result += f"- Account ID: {acc.account_id} | Balance: £{acc.account_balance}\n"
            _res = f"{result}\n\nDATA SUMMARY:\n{result}"
            logger.info(f"Tool returned: {str(_res)[:1000]}")
            return _res
    except Exception as e:
        _res = f"Database error: {str(e)}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res
