from prefect import flow, task, get_run_logger
from schemas.api_schema import OnboardingLLMResult
from models.database_models import User
from config import SessionLocal
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

@task(retries=3, retry_delay_seconds=30)
def extract_persona_task(form_context: str) -> dict:
    logger = get_run_logger()
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
    
    llm_analysis_dict = chain.invoke({"form_data": form_context})
    return llm_analysis_dict

@task
def commit_onboarding_db_task(user_uuid: str, llm_analysis_dict: dict):
    logger = get_run_logger()
    llm_analysis = OnboardingLLMResult(**llm_analysis_dict)
    logger.info(f"Deduced persona: {llm_analysis.persona}")
    
    with SessionLocal() as db:
        user = db.query(User).filter(User.user_uuid == user_uuid).first()
        if user:
            user.persona = llm_analysis.persona
            user.is_onboarded = True
            db.commit()
            logger.info("Successfully updated user profile in database.")
        else:
            logger.error(f"User {user_uuid} not found during onboarding commit.")

@flow(name="Process User Onboarding")
def process_user_onboarding_flow(user_uuid: str, goals: list, income_pattern: list, liabilities: list, user_summary: str):
    logger = get_run_logger()
    from services.llm_manager import GlobalLLMManager
    GlobalLLMManager.try_acquire()
    
    try:
        logger.info(f"Processing static onboarding completion in background for {user_uuid}")
        
        form_context = f"""
        User Form Submission:
        - Selected Goals: {goals}
        - Income Pattern: {income_pattern}
        - Liabilities: {liabilities}
        
        User's Personal Summary:
        "{user_summary}"
        """
        
        llm_analysis_dict = extract_persona_task(form_context)
        commit_onboarding_db_task(user_uuid, llm_analysis_dict)
        
    except Exception as e:
        logger.error(f"Onboarding background flow failed: {e}")
        raise e
    finally:
        GlobalLLMManager.release()
