from langchain_core.tools import tool
from config import SessionLocal
from models.database_models import Account
from pydantic import BaseModel, Field

class GetConnectedAccountsInput(BaseModel):
    user_uuid: str = Field(..., description="The user UUID to query accounts for.")

@tool(args_schema=GetConnectedAccountsInput)
def get_connected_accounts(user_uuid: str) -> str:
    """Use this tool to retrieve a list of all connected accounts and their account IDs for the user."""
    try:
        with SessionLocal() as session:
            accounts = session.query(Account).filter_by(user_uuid=user_uuid).all()
            if not accounts:
                return "No connected accounts found."
            
            result = "Connected Accounts:\n"
            for acc in accounts:
                result += f"- Account ID: {acc.account_id} | Balance: £{acc.account_balance}\n"
            return f"{result}\n\nDATA SUMMARY:\n{result}"
    except Exception as e:
        return f"Database error: {str(e)}"
