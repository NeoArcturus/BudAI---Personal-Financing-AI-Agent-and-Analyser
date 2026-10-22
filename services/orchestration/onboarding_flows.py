from langfuse.langchain import CallbackHandler
from prefect import flow, task, get_run_logger
from schemas.api_schema import OnboardingLLMResult
from models.database_models import User, Bucket, BucketType, UserLifestyleProfile
from config import SessionLocal
import os
import uuid
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from services.logger_setup import get_core_logger, log_pipeline_data

logger = get_core_logger(__name__)

def deduce_deterministic_persona(goals: list, income_pattern: list, liabilities: list, user_summary: str = "") -> str:
    """
    Deterministic rule-based persona deduction for non-LLM resilience.
    Guarantees user onboarding completes and core dashboard works even if LLM is offline.
    """
    summary_lower = (user_summary or "").lower()
    goals_str = " ".join(goals or []).lower()
    income_str = " ".join(income_pattern or []).lower()
    liabilities_str = " ".join(liabilities or []).lower()
    combined = f"{goals_str} {income_str} {liabilities_str} {summary_lower}"

    if any(k in combined for k in ["student", "tuition", "university", "college", "student_loans", "academic"]):
        return "STUDENT"
    if any(k in combined for k in ["artist", "creative", "musician", "designer", "creator", "writer", "photographer", "illustrator"]):
        return "CREATIVE"
    if any(k in combined for k in ["retire", "pension", "superannuation", "senior"]):
        return "RETIREE"
    if any(k in combined for k in ["business", "freelance_management", "corporate", "revenue", "company", "founder", "entrepreneur", "shop", "store", "commercial", "freelance", "bakery", "enterprise"]):
        return "BUSINESS"
    return "PROFESSIONAL"

@task(retries=1, retry_delay_seconds=3)
def extract_persona_task(form_context: str, fallback_persona: str = "PROFESSIONAL") -> dict:
    try:
        run_logger = get_run_logger()
    except Exception:
        run_logger = logger

    try:
        OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
        if not OLLAMA_BASE_URL.endswith("/v1"):
            OLLAMA_BASE_URL = f"{OLLAMA_BASE_URL}/v1"

        onboarding_llm = ChatOpenAI(
            model="Qwen3.5-9B-GGUF",
            base_url=OLLAMA_BASE_URL,
            api_key="budai-local",
            temperature=0,
            streaming=False,
            max_tokens=1000,
            timeout=15  # Fail fast to prevent user stalls when LLM is offline
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
        run_logger.info("Invoking LLM for onboarding extraction...")
        
        llm_analysis_dict = chain.invoke(
            {"form_data": form_context}, 
            config={"callbacks": [CallbackHandler()], "metadata": {"langfuse_tags": ["onboarding"]}}
        )
        return llm_analysis_dict
    except Exception as e:
        run_logger.warning(f"LLM offline or timed out during onboarding extraction: {e}. Utilizing deterministic fallback.")
        return {
            "persona": fallback_persona,
            "keypoints": [
                "Profile initialized with deterministic baseline rules.",
                "Non-LLM shadow ledger and accounts dashboard fully active.",
                "AI contextual analysis will refresh once LLM service resumes."
            ]
        }

@task
def commit_onboarding_db_task(user_uuid: str, llm_analysis_dict: dict):
    try:
        run_logger = get_run_logger()
    except Exception:
        run_logger = logger

    persona_deduced = llm_analysis_dict.get("persona", "PROFESSIONAL")
    run_logger.info(f"Deduced persona: {persona_deduced}")
    
    with SessionLocal() as db:
        user = db.query(User).filter(User.user_uuid == user_uuid).first()
        if user:
            user.persona = persona_deduced
            user.is_onboarded = True
            
            # Seed the mandatory DEFAULT bucket for the zero-sum ledger system
            existing_bucket = db.query(Bucket).filter(Bucket.user_id == user_uuid, Bucket.type == BucketType.DEFAULT).first()
            if not existing_bucket:
                default_bucket = Bucket(
                    id=str(uuid.uuid4()),
                    user_id=user_uuid,
                    type=BucketType.DEFAULT,
                    name="Unallocated Funds",
                    priority_index=100.0,
                    cached_balance=0.0,
                    is_active=True
                )
                db.add(default_bucket)
                run_logger.info("Seeded default Unallocated Funds bucket.")

            # Update or create UserLifestyleProfile
            profile = db.query(UserLifestyleProfile).filter(UserLifestyleProfile.user_uuid == user_uuid).first()
            if not profile:
                profile = UserLifestyleProfile(
                    profile_uuid=str(uuid.uuid4()),
                    user_uuid=user_uuid,
                    macro_persona=persona_deduced
                )
                db.add(profile)
            else:
                profile.macro_persona = persona_deduced

            db.commit()
            log_pipeline_data(
                logger=logger,
                pipeline="onboarding",
                stage="db_commit_complete",
                data={
                    "user_uuid": user_uuid,
                    "persona": persona_deduced,
                    "is_onboarded": True
                },
                message=f"Successfully committed onboarding profile for user {user_uuid}"
            )
        else:
            run_logger.error(f"User {user_uuid} not found during onboarding commit.")

@flow(name="Process User Onboarding")
def process_user_onboarding_flow(user_uuid: str, goals: list, income_pattern: list, liabilities: list, user_summary: str):
    try:
        run_logger = get_run_logger()
    except Exception:
        run_logger = logger

    from services.llm_manager import GlobalLLMManager
    acquired = False
    try:
        GlobalLLMManager.acquire_background()
        acquired = True
    except Exception as e:
        run_logger.warning(f"GlobalLLMManager acquire bypassed: {e}")
    
    try:
        run_logger.info(f"Processing onboarding completion in background for {user_uuid}")
        fallback_persona = deduce_deterministic_persona(goals, income_pattern, liabilities, user_summary)
        
        form_context = f"""
        User Form Submission:
        - Selected Goals: {goals}
        - Income Pattern: {income_pattern}
        - Liabilities: {liabilities}
        
        User's Personal Summary:
        "{user_summary}"
        """
        
        llm_analysis_dict = extract_persona_task(form_context, fallback_persona=fallback_persona)
        commit_onboarding_db_task(user_uuid, llm_analysis_dict)
        
    except Exception as e:
        run_logger.error(f"Onboarding background flow failed: {e}")
    finally:
        if acquired:
            try:
                GlobalLLMManager.release()
            except Exception:
                pass

def run_onboarding_flow_direct(user_uuid: str, goals: list, income_pattern: list, liabilities: list, user_summary: str):
    """Fallback runner for direct in-process execution when Prefect daemon is unreachable."""
    try:
        fallback_persona = deduce_deterministic_persona(goals, income_pattern, liabilities, user_summary)
        form_context = f"""
        User Form Submission:
        - Selected Goals: {goals}
        - Income Pattern: {income_pattern}
        - Liabilities: {liabilities}
        
        User's Personal Summary:
        "{user_summary}"
        """
        # Execute tasks directly using .fn attribute
        llm_analysis_dict = extract_persona_task.fn(form_context, fallback_persona=fallback_persona)
        commit_onboarding_db_task.fn(user_uuid, llm_analysis_dict)
    except Exception as e:
        logger.error(f"Direct onboarding fallback flow failed: {e}", exc_info=True)
