from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from models.database_models import User
from middleware.auth_middleware import get_current_user
from services.logger_setup import get_core_logger
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")

logger = get_core_logger(__name__)

router = APIRouter(prefix="/api/chat", tags=["NLP Intent"])

class IntentRequest(BaseModel):
    query: str

class ParsedIntent(BaseModel):
    time_type: Optional[str] = Field(None, description="'daily', 'weekly', or 'monthly'")
    from_date: Optional[str] = Field(None, description="ISO date format YYYY-MM-DD")
    to_date: Optional[str] = Field(None, description="ISO date format YYYY-MM-DD")
    category: Optional[str] = Field(None, description="Specific spending category if mentioned")
    account_ids: Optional[List[str]] = Field(None, description="List of account IDs if specific accounts mentioned")

@router.post("/parse-intent", response_model=ParsedIntent)
async def parse_dashboard_intent(request: IntentRequest, current_user: User = Depends(get_current_user)):
    """
    Parses natural language into strict JSON parameters to filter the dashboard widgets.
    """
    try:
        llm = ChatOpenAI(
            model="Qwen3.5-9B-GGUF", # Mac: model="mlx-community/Qwen3.5-4B-4bit", 
            base_url=OLLAMA_BASE_URL,
            api_key="budai-local",
            temperature=0
        ).bind(response_format={"type": "json_object"})
        
        prompt = PromptTemplate.from_template(
            "You are an intent parser for a financial dashboard. "
            "Extract the filtering parameters from the user's query.\n"
            "Return ONLY a JSON object matching this schema:\n"
            "{{\n"
            "  \"time_type\": \"'daily', 'weekly', or 'monthly' (default to monthly)\",\n"
            "  \"from_date\": \"YYYY-MM-DD (calculate relative to today if needed)\",\n"
            "  \"to_date\": \"YYYY-MM-DD\",\n"
            "  \"category\": \"Specific category (e.g., 'Groceries', 'Transport') or null\",\n"
            "  \"account_ids\": specific account IDs mentioned, or null\n"
            "}}\n\n"
            "User Query: {query}"
        )
        
        chain = prompt | llm
        result = chain.invoke({"query": request.query})
        
        parsed_json = json.loads(result.content)
        return ParsedIntent(**parsed_json)
    except Exception as e:
        logger.error(json.dumps({"message": f"Error parsing intent: {e}", "status_code": 500}))
        raise HTTPException(status_code=500, detail="Failed to parse intent")
