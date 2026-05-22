from typing import TypedDict, Dict, Any, List, Optional
from langchain_core.messages import BaseMessage
from services.logger_setup import get_core_logger
logger = get_core_logger(__name__)

class BudAIState(TypedDict, total=False):
    user_uuid: str
    session_id: Optional[str]
    active_account_id: str
    user_input: str
    chat_history: List[BaseMessage]
    selected_worker: Optional[str]
    worker_summary: Optional[str]
    cache_id: Optional[str]
    chart_type: Optional[str]
    ui_trigger_tag: Optional[str]
    final_response: str
    raw_data: Optional[Any]
    is_explanation: Optional[bool]

