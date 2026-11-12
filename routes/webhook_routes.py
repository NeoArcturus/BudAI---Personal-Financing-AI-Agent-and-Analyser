import json
from fastapi import APIRouter, Request, BackgroundTasks
from controllers.webhook_controller import handle_truelayer_webhook
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
        target_acc_id = acc_id or payload.get("account_id")
        return await handle_truelayer_webhook(payload, background_tasks, user_uuid, bank_uuid, target_acc_id)
    except Exception as e:
        logger.error(json.dumps({"message": f"Failed to process TrueLayer webhook: {e}", "status_code": 500}))
        return {"status": "error", "message": "Failed to parse webhook"}
