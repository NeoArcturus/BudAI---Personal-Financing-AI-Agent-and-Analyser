from fastapi import APIRouter, Depends, BackgroundTasks
from middleware.auth_middleware import get_current_user
from models.database_models import User
from schemas.api_schema import OnboardingFormRequest
from controllers.onboarding.post import complete_onboarding

router = APIRouter(prefix="/api/onboarding", tags=["onboarding"])

@router.post("/complete")
async def complete_onboarding_route(request: OnboardingFormRequest, current_user: User = Depends(get_current_user)):
    return await complete_onboarding(request, current_user)
