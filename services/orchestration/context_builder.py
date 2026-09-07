import json
from datetime import datetime
from sqlmodel import Session, select
from models.database_models import SystemAlert
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

def build_evaluation_context(session: Session, user_uuid: str, masked_state: list, discoveries: dict, trigger_event: dict = None) -> str:
    """
    Step 3 of Phase 3: Context Hydration
    Compiles the masked state, mathematical discoveries, and strict behavioral guardrails 
    into a rigid, un-hallucinatable text prompt for the LLM.
    """
    logger.info(json.dumps({"message": f"Compiling agent brief for {user_uuid}", "status_code": 200}))
    
    # 1. State Block (Masked)
    state_block = "CURRENT STATE (Masked):\n"
    for b in masked_state:
        target = f" | Target: £{b['target_amount']:.2f}" if b['target_amount'] else ""
        state_block += f"- {b['alias']} ({b['type']}): £{b['balance']:.2f}{target}\n"
        
    # 2. Event Block
    event_block = "TRIGGER EVENT:\n"
    if trigger_event:
        event_block += json.dumps(trigger_event, indent=2) + "\n"
    else:
        event_block += "Scheduled background evaluation.\n"
        
    # 3. Discovery Block (From Math Pipelines)
    discovery_block = "NEW DISCOVERIES:\n"
    subs = discoveries.get("new_subscriptions", [])
    utils = discoveries.get("new_utilities", [])
    if not subs and not utils:
        discovery_block += "None.\n"
    if subs:
        discovery_block += f"- Subscriptions: {json.dumps(subs)}\n"
    if utils:
        discovery_block += f"- Utilities: {json.dumps(utils)}\n"
        
    # 3.5 System Warnings (Broken Connections)
    from models.database_models import Bank
    from models.status_codes import OpenBankingStatus
    broken_banks = session.execute(
        select(Bank)
        .where(Bank.user_uuid == user_uuid)
        .where(Bank.consent_status.in_([OpenBankingStatus.CONNECTION_EXPIRED.value, OpenBankingStatus.BANK_REVOKED_CONSENT.value]))
    ).scalars().all()
    
    warning_block = ""
    if broken_banks:
        warning_block = "SYSTEM WARNINGS:\n"
        for bank in broken_banks:
            warning_block += f"- CRITICAL: Your connection to '{bank.bank_name}' has expired or been revoked. Live synchronization is currently broken. Prompt user to re-authenticate immediately.\n"
        warning_block += "\n"
        
    # 4. Throttle Block (Guardrails via DB)
    today = datetime.utcnow().date()
    alerts_today = session.execute(
        select(SystemAlert)
        .where(SystemAlert.user_id == user_uuid)
        .where(SystemAlert.timestamp >= today)
    ).all()
    alert_count = len(alerts_today)
    
    throttle_block = "GUARDRAILS:\n"
    throttle_block += f"- You have sent {alert_count} alerts today. Hard limit is 3.\n"
    throttle_block += "- You are physically sandboxed. Do not hallucinate PII.\n"
    
    # 5. Instruction Lock
    instruction_block = """
INSTRUCTIONS:
You are the BudAI Autonomous Agent. Evaluate the state and discoveries above. You must act proactively by invoking your native tools:
1. NEW DISCOVERIES: If you see a new subscription or utility, use the create_bucket tool to build a dedicated bucket for it. Then, use the execute_virtual_transfer tool to move enough funds from the DEFAULT bucket into this new bucket so the upcoming bill is covered.
2. CAPITAL ALLOCATION: If you observe that the DEFAULT bucket has a large surplus, but a TARGET_DATE or ACCUMULATING bucket is falling behind its target, proactively use the execute_virtual_transfer tool to move excess capital into the savings buckets.
3. ALERTS: If you create a new bucket or move a significant amount of money, use the alert tool to push a notification explaining what you did and why.
4. If no proactive action is required, you must invoke the sleep tool.
"""
    
    evaluation_context = f"{state_block}\n{event_block}\n{discovery_block}\n{warning_block}{throttle_block}\n{instruction_block}"
    
    return evaluation_context
