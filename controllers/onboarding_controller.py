import uuid
import threading
from models.database_models import User, Bucket, BucketType, UserLifestyleProfile
from schemas.api_schema import OnboardingFormRequest
from services.logger_setup import get_core_logger, log_pipeline_data
from services.orchestration.onboarding_flows import deduce_deterministic_persona, run_onboarding_flow_direct
from services.orchestration.delta_sync import perform_physical_delta_sync
from config import SessionLocal
from prefect.client.orchestration import get_client
from prefect.exceptions import ObjectNotFound, PrefectHTTPStatusError, ObjectLimitReached

logger = get_core_logger(__name__)


def complete_core_onboarding_sync(
    user_uuid: str,
    goals: list,
    income_pattern: list,
    liabilities: list,
    user_summary: str = "",
    name: str = None,
    date_of_birth: str = None,
    employment_status: str = None,
    country_of_tax_residence: str = None
) -> str:
    """
    Synchronously provisions the core zero-sum primitives.
    Guarantees user.is_onboarded = True and UNALLOCATED bucket exists.
    """
    persona = deduce_deterministic_persona(goals, income_pattern, liabilities, user_summary)
    with SessionLocal() as session:
        user = session.query(User).filter(User.user_uuid == user_uuid).first()
        if not user:
            raise ValueError(f"User {user_uuid} not found")

        user.is_onboarded = True
        if not user.persona:
            user.persona = persona
        if name and not user.name:
            user.name = name
        if employment_status:
            user.employment_status = employment_status
        if country_of_tax_residence:
            user.country_of_tax_residence = country_of_tax_residence
        if date_of_birth:
            from datetime import datetime
            try:
                user.date_of_birth = datetime.strptime(date_of_birth, "%Y-%m-%d")
            except Exception:
                pass

        # Seed mandatory UNALLOCATED bucket
        default_bucket = session.query(Bucket).filter(
            Bucket.user_id == user_uuid,
            Bucket.type == BucketType.UNALLOCATED
        ).first()

        if not default_bucket:
            default_bucket = Bucket(
                id=str(uuid.uuid4()),
                user_id=user_uuid,
                type=BucketType.UNALLOCATED,
                name="Unallocated",
                cached_balance=0.0,
                cached_balances={},
                currency="GBP",
                is_active=True
            )
            session.add(default_bucket)

        # Ensure UserLifestyleProfile exists
        profile = session.query(UserLifestyleProfile).filter(
            UserLifestyleProfile.user_uuid == user_uuid
        ).first()
        if not profile:
            profile = UserLifestyleProfile(
                profile_uuid=str(uuid.uuid4()),
                user_uuid=user_uuid,
                macro_persona=persona
            )
            session.add(profile)

        session.commit()

        try:
            perform_physical_delta_sync(session, user_uuid)
        except Exception as se:
            logger.warning(f"Initial delta sync for {user_uuid} deferred: {se}")

        log_pipeline_data(
            logger=logger,
            pipeline="onboarding",
            stage="core_provisioning",
            data={
                "user_uuid": user_uuid,
                "is_onboarded": True,
                "persona": user.persona,
                "default_bucket_id": default_bucket.id
            },
            message=f"Core onboarding completed for {user_uuid}. Unallocated bucket & persona initialized."
        )
        return user.persona


async def complete_onboarding(request: OnboardingFormRequest, current_user: User):
    """Handles onboarding completion request."""
    logger.info(f"Received onboarding complete request for {current_user.user_uuid}.")

    try:
        persona = complete_core_onboarding_sync(
            user_uuid=current_user.user_uuid,
            goals=request.goals,
            income_pattern=request.income_pattern,
            liabilities=request.liabilities,
            user_summary=request.user_summary,
            name=request.name,
            date_of_birth=request.date_of_birth,
            employment_status=request.employment_status,
            country_of_tax_residence=request.country_of_tax_residence
        )
    except Exception as e:
        logger.error(f"Core onboarding provisioning failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Core onboarding failed: {e}"
        }

    try:
        async with get_client() as client:
            deployment = await client.read_deployment_by_name("Process User Onboarding/cloud-onboarding")
            await client.create_flow_run_from_deployment(
                deployment_id=deployment.id,
                parameters={
                    "user_uuid": current_user.user_uuid,
                    "goals": request.goals,
                    "income_pattern": request.income_pattern,
                    "liabilities": request.liabilities,
                    "user_summary": request.user_summary
                }
            )
            logger.info(f"Enqueued Prefect background onboarding refinement for {current_user.user_uuid}")
    except (ObjectNotFound, ObjectLimitReached, PrefectHTTPStatusError, Exception) as e:
        logger.warning(f"Prefect deployment unavailable ({type(e).__name__}: {e}). Spawning local non-blocking thread for AI enrichment.")
        threading.Thread(
            target=run_onboarding_flow_direct,
            args=(current_user.user_uuid, request.goals, request.income_pattern, request.liabilities, request.user_summary),
            daemon=True
        ).start()

    return {
        "status": "success",
        "is_onboarded": True,
        "persona": persona,
        "message": "Onboarding completed successfully. Core dashboard services are active."
    }
