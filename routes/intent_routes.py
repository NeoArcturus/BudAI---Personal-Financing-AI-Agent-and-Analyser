from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import json
import re
from models.database_models import User
from middleware.auth_middleware import get_current_user
from services.logger_setup import get_core_logger
from services.llm_factory import get_chat_llm

logger = get_core_logger(__name__)

router = APIRouter(prefix="/api/chat", tags=["NLP Intent"])

class IntentRequest(BaseModel):
    query: str

class ParsedIntent(BaseModel):
    time_type: Optional[str] = Field("monthly", description="'daily', 'weekly', or 'monthly'")
    from_date: Optional[str] = Field(None, description="ISO date format YYYY-MM-DD")
    to_date: Optional[str] = Field(None, description="ISO date format YYYY-MM-DD")
    category: Optional[str] = Field(None, description="Specific spending category if mentioned")
    account_ids: Optional[List[str]] = Field(None, description="List of account IDs if specific accounts mentioned")

def fallback_intent_extraction(query: str) -> ParsedIntent:
    """
    Deterministic rule-based intent fallback when local LLM is offline or output is corrupted.
    """
    q = query.lower()
    time_type = "monthly"
    if "daily" in q or "today" in q or "yesterday" in q:
        time_type = "daily"
    elif "week" in q or "weekly" in q:
        time_type = "weekly"

    category = None
    common_cats = [
        "dining", "food", "groceries", "transport", "utilities", 
        "shopping", "entertainment", "bills", "income", "travel"
    ]
    for c in common_cats:
        if c in q:
            category = c.title()
            break

    return ParsedIntent(time_type=time_type, category=category)

@router.post("/parse-intent", response_model=ParsedIntent)
async def parse_dashboard_intent(request: IntentRequest, current_user: User = Depends(get_current_user)):
    """
    Parses natural language into strict JSON parameters to filter the dashboard widgets.
    Uses structured output with regex sanitization and deterministic fallback.
    """
    if not request.query or not request.query.strip():
        return ParsedIntent(time_type="monthly")

    try:
        llm = get_chat_llm(
            temperature=0,
            timeout=15.0
        )
        
        prompt = (
            "You are an intent parser for a financial dashboard. "
            "Extract the filtering parameters from the user's query.\n"
            "Return ONLY a valid JSON object matching this schema:\n"
            "{\n"
            '  "time_type": "\'daily\', \'weekly\', or \'monthly\'",\n'
            '  "from_date": "YYYY-MM-DD or null",\n'
            '  "to_date": "YYYY-MM-DD or null",\n'
            '  "category": "Specific category (e.g., \'Food & Dining\', \'Transport\') or null",\n'
            '  "account_ids": null\n'
            "}\n\n"
            f"User Query: {request.query}"
        )

        res = await llm.ainvoke(prompt)
        content = res.content if isinstance(res.content, str) else str(res.content)
        
        # Extract JSON from potential markdown wrapping
        json_match = re.search(r"\{.*?\}", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            return ParsedIntent(**parsed)
        
        return fallback_intent_extraction(request.query)

    except Exception as e:
        logger.warning(f"LLM intent parsing failed, using deterministic fallback: {e}")
        return fallback_intent_extraction(request.query)
