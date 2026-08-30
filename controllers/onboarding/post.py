from models.database_models import User
from schemas.api_schema import OnboardingFormRequest
from services.logger_setup import get_core_logger
from prefect.client.orchestration import get_client

from prefect.exceptions import ObjectNotFound, PrefectHTTPStatusError, ObjectLimitReached

logger = get_core_logger(__name__)

async def complete_onboarding(request: OnboardingFormRequest, current_user: User):
    logger.info(f"Received onboarding complete request for {current_user.user_uuid}. Queuing background task via Prefect.")
    
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
    except ObjectNotFound:
        logger.error("Prefect deployment 'Process User Onboarding/main-deployment' not found. Ensure the serve_flows.py worker is running.")
    except ObjectLimitReached:
        logger.error("Prefect object limit reached (e.g., too many flow runs or deployments for the current plan).")
    except PrefectHTTPStatusError as e:
        logger.error(f"Prefect API rejected the request (likely a plan limitation or auth issue). Details: {e}")
    except Exception as e:
        logger.error(f"Unexpected error triggering Prefect deployment: {e}")
    
    return {
        "status": "processing",
        "message": "AI analysis started in the background."
    }
