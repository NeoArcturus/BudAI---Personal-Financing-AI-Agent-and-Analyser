from fastapi import APIRouter, Request, BackgroundTasks
from controllers.webhooks.post import handle_truelayer_webhook
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

router = APIRouter(prefix="/api/webhooks", tags=["Webhooks"])

@router.post("/truelayer")
async def truelayer_webhook_route(
    request: Request, 
    background_tasks: BackgroundTasks,
    user_uuid: str = None, 
    bank_uuid: str = None, 
    acc_id: str = None
):
    try:
        payload = await request.json()
        return await handle_truelayer_webhook(payload, background_tasks, user_uuid, bank_uuid, acc_id)
    except Exception as e:
        logger.error(f"Failed to process TrueLayer webhook: {e}")
        return {"status": "error", "message": "Failed to parse webhook"}
