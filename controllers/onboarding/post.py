import json
from fastapi import APIRouter, Depends, BackgroundTasks
from middleware.auth_middleware import get_current_user
from models.database_models import User
from schemas.api_schema import OnboardingFormRequest, OnboardingLLMResult
from services.logger_setup import get_core_logger
from config import SessionLocal
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

import os
from langchain_openai import ChatOpenAI

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
if not OLLAMA_BASE_URL.endswith("/v1"):
    OLLAMA_BASE_URL = f"{OLLAMA_BASE_URL}/v1"

onboarding_llm = ChatOpenAI(
    model="lmstudio-community/Qwen3.5-9B-GGUF",
    base_url=OLLAMA_BASE_URL,
    api_key="budai-local",
    temperature=0,
    streaming=False,
    max_tokens=2000,
    timeout=600
)
logger = get_core_logger(__name__)

async def _process_llm_onboarding_background(user_uuid: str, request: OnboardingFormRequest):
    from services.llm_manager import GlobalLLMManager
    GlobalLLMManager.try_acquire()
    
    try:
        logger.info(f"Processing static onboarding completion in background for {user_uuid}")
        
        # 1. Compile the hardcoded answers and user summary into a prompt string
        form_context = f"""
        User Form Submission:
        - Selected Goals: {request.goals}
        - Income Pattern: {request.income_pattern}
        - Liabilities: {request.liabilities}
        
        User's Personal Summary:
        "{request.user_summary}"
        """
        
        parser = JsonOutputParser(pydantic_object=OnboardingLLMResult)
        
        prompt = PromptTemplate(
            template="""
You are BudAI's core analytical engine.
Analyze the user's hardcoded form answers and their personal summary.
1. Deduce their macro Persona (MUST be one of: STUDENT, PROFESSIONAL, BUSINESS, RETIREE, CREATIVE).
2. Extract key financial context points (3-4 synthesized sentences).

{format_instructions}

Data:
{form_data}
""",
            input_variables=["form_data"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        
        chain = prompt | onboarding_llm | parser
        
        logger.info("Invoking LLM for onboarding extraction...")
        llm_analysis_dict = await chain.ainvoke({"form_data": form_context})
        
        # Validate through Pydantic
        llm_analysis = OnboardingLLMResult(**llm_analysis_dict)
        
        logger.info(f"Deduced persona: {llm_analysis.persona}")
        
        # 3. Definitive Widget Mapping based on Persona
        PERSONA_WIDGET_MAP = {
            "STUDENT": ["cashFlow", "expenseDistribution", "analyticsHabits", "aiChat"],
            "PROFESSIONAL": ["spendingTrend", "analyticsSubscriptions", "analyticsHealth", "aiChat"],
            "BUSINESS": ["cashFlow", "ledger", "analyticsAnomalies", "financialNews"],
            "RETIREE": ["commodityMarket", "analyticsRisk", "expenseDistribution", "aiChat"],
            "CREATIVE": ["spendingTrend", "analyticsHabits", "analyticsAnomalies", "aiChat"]
        }
        
        # Fallback to PROFESSIONAL if LLM hallucinates the persona
        definitive_widgets = PERSONA_WIDGET_MAP.get(llm_analysis.persona, PERSONA_WIDGET_MAP["PROFESSIONAL"])
        logger.info(f"Assigned definitive widgets (logging only, dynamic mapping used in frontend): {definitive_widgets}")
        
        # 4. Update Database safely with a new session
        with SessionLocal() as db:
            user = db.query(User).filter(User.user_uuid == user_uuid).first()
            if user:
                user.persona = llm_analysis.persona
                user.is_onboarded = True
                db.commit()
                logger.info("Successfully updated user profile in database.")
            
    except Exception as e:
        logger.error(f"Onboarding background completion failed: {e}")
    finally:
        from services.llm_manager import GlobalLLMManager
        GlobalLLMManager.release()

async def complete_onboarding(request: OnboardingFormRequest, background_tasks: BackgroundTasks, current_user: User):
    logger.info(f"Received onboarding complete request for {current_user.user_uuid}. Queuing background task.")
    
    # Kick off the background task
    background_tasks.add_task(_process_llm_onboarding_background, current_user.user_uuid, request)
    
    # Return immediately
    return {
        "status": "processing",
        "message": "AI analysis started in the background."
    }
