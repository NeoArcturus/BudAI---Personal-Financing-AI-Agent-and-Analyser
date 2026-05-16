from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
from middleware.auth_middleware import get_current_user
from models.database_models import User
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from services.profile_builder import ProfileBuilder
import logging

logger = logging.getLogger("uvicorn.error")

advisor_router = APIRouter(prefix="/api/advisor", tags=["advisor"])

class SummarizeRequest(BaseModel):
    widget_id: str
    context_data: Any

@advisor_router.post("/summarize")
async def summarize_data(request: SummarizeRequest, current_user: User = Depends(get_current_user)):
    try:
        profile_builder = ProfileBuilder(current_user.user_uuid)
        mrfp = profile_builder.build_profile()

        llm = ChatOllama(
            model="qwen3:4b",
            temperature=0.2,
            keep_alive=300
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are BudAI, a precise senior financial advisor. 
Below is the user's exact, mathematically verified Multi-Resolution Financial Profile.

<USER_PROFILE>
{mrfp}
</USER_PROFILE>

CRITICAL RULES FOR HALLUCINATION PREVENTION:
1. You may ONLY quote numbers, merchants, and dates that appear exactly inside the <USER_PROFILE> block or the specific data provided.
2. Provide exactly 2-3 sentences of actionable insight. Use GBP currency symbols (£). Do not use emojis.
3. Do not round numbers. Use exact GBP amounts.
4. If predicting a future shortage, cite the specific upcoming bill from Tier 2 that will cause it.
"""),
            ("human", "Widget: {widget_id}\nData: {data}")
        ])

        chain = prompt | llm

        result = await chain.ainvoke({
            "widget_id": request.widget_id,
            "data": str(request.context_data)
        })

        insight = result.content.strip()
        
        # Ensure it's not too long, though the prompt should handle it
        return {"insight": insight}
    except Exception as e:
        logger.error(f"Advisor summarization failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate insight.")
