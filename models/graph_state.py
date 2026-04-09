from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage


class BudAIState(TypedDict):
    user_uuid: str
    active_account_id: str
    user_input: str
    chat_history: List[BaseMessage]
    selected_worker: Optional[str]
    worker_summary: Optional[str]
    cache_id: Optional[str]
    chart_type: Optional[str]
    ui_trigger_tag: Optional[str]
    final_response: str
