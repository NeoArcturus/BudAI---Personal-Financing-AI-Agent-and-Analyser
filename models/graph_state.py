from typing import TypedDict, Dict, Any, List, Optional, Annotated
from langchain_core.messages import BaseMessage
from services.logger_setup import get_core_logger
logger = get_core_logger(__name__)

def update_latest(old: Optional[str], new: Optional[str]) -> Optional[str]:
    return new if new is not None else old

class BudAIState(TypedDict, total=False):
    user_uuid: str
    session_id: Optional[str]
    active_account_id: str
    user_input: str
    chat_history: List[BaseMessage]
    selected_worker: Optional[str]
    worker_summary: Optional[str]
    cache_id: Annotated[Optional[str], update_latest]
    chart_type: Annotated[Optional[str], update_latest]
    ui_trigger_tag: Annotated[Optional[str], update_latest]
    final_response: str
    raw_data: Optional[Any]
    is_explanation: Optional[bool]

