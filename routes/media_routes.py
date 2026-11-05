from fastapi import APIRouter, Depends, Response
from middleware.auth_middleware import get_current_user
from models.database_models import User
from controllers.media.post import execute_tool, MediaExecuteRequest

router = APIRouter(prefix="/api/media", tags=["media"])

@router.post('/execute')
async def execute_tool_route(
    request: MediaExecuteRequest, 
    response: Response,
    current_user: User = Depends(get_current_user)
):
    response.headers["Deprecation"] = "true"
    response.headers["Link"] = '</api/widgets>; rel="successor-version"'
    return await execute_tool(request, current_user)
