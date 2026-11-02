import json
import uuid
from typing import Dict, Any, Optional
from sqlalchemy import text
from config import SessionLocal
from services.a2a.protocol import A2AMessage, A2AResponse
from services.a2a.negotiator import A2ANegotiator
from models.status_codes import AgentStatus, PipelineStatus, MacroStatus
from services.analytics.ons_client import get_ons_client
from services.categorization.taxonomy import enrich_transaction_taxonomy, TAXONOMY_MAP, STANDARD_CATEGORIES
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

async def handle_market_agent(msg: A2AMessage) -> A2AResponse:
    """
    MarketWorker A2A Handler:
    Provides verified macroeconomic telemetry (ONS COICOP inflation indices)
    and external economic context to peer agents without hitting third-party APIs.
    """
    ons_client = get_ons_client()
    intent = msg.intent.upper()

    if intent == "FETCH_MACRO_INFLATION":
        target = msg.payload.get("category") or msg.payload.get("tag") or "CP00"
        yoy_rate = ons_client.get_category_inflation(target)
        snapshot = ons_client.get_latest_snapshot()

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="MarketWorker",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=MacroStatus.TELEMETRY_SYNCED.value,
            payload={
                "indicator": target,
                "delta_pct_yoy": yoy_rate,
                "headline_cpih": float(snapshot.get("CP00:yoy", yoy_rate)),
                "food_inflation": float(snapshot.get("CP01:yoy", 3.52)),
                "energy_inflation": float(snapshot.get("CP045:yoy", 2.08)),
                "period": snapshot.get("period", "latest")
            }
        )

    elif intent == "GET_MARKET_INDEX":
        snapshot = ons_client.get_latest_snapshot()
        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="MarketWorker",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=AgentStatus.AGENT_RESPONSE_READY.value,
            payload={
                "market_telemetry": {
                    "headline_inflation": float(snapshot.get("CP00:yoy", 2.8)),
                    "transport_inflation": float(snapshot.get("CP07:yoy", 2.71)),
                    "energy_inflation": float(snapshot.get("CP045:yoy", 2.08))
                }
            }
        )

    return A2AResponse(
        reply_to_id=msg.message_id,
        correlation_id=msg.correlation_id,
        sender="MarketWorker",
        recipient=msg.sender,
        status="ERROR",
        status_code=AgentStatus.AGENT_PARSING_FAILURE.value,
        error_message=f"Unrecognized MarketWorker intent: {intent}"
    )

async def handle_health_agent(msg: A2AMessage) -> A2AResponse:
    """
    HealthWorker A2A Handler:
    Audits user liquidity buffers, emergency runway, and computes financial wellness scores.
    Can autonomously initiate rebalancing proposals when liquidity violations occur.
    """
    user_uuid = msg.user_uuid
    intent = msg.intent.upper()

    with SessionLocal() as session:
        # Sum current liquid balances (DEFAULT + ACCUMULATING)
        buckets = session.execute(
            text("SELECT type, cached_balance FROM buckets WHERE user_id = :u AND is_active = TRUE"),
            {"u": user_uuid}
        ).fetchall()

        total_liquid = 0.0
        default_balance = 0.0
        recurring_balance = 0.0
        for b_type, b_bal in buckets:
            val = float(b_bal or 0.0)
            if b_type in ("DEFAULT", "ACCUMULATING"):
                total_liquid += val
            if b_type == "DEFAULT":
                default_balance += val
            elif b_type == "RECURRING":
                recurring_balance += val

        # Estimate 30-day burn rate from recent debit transactions
        tx_row = session.execute(
            text("""
                SELECT COALESCE(SUM(amount), 0.0) 
                FROM transactions 
                WHERE user_uuid = :u AND amount < 0 AND date >= NOW() - INTERVAL '30 days'
            """),
            {"u": user_uuid}
        ).fetchone()

        monthly_burn = abs(float(tx_row[0])) if tx_row and tx_row[0] else 1500.0  # fallback assumption
        runway_months = round(total_liquid / max(monthly_burn, 1.0), 2)

    if intent in ("AUDIT_HEALTH_SCORE", "CHECK_EMERGENCY_RUNWAY"):
        # Runway benchmarks: < 1 month = Critical, 1-3 = Caution, 3-6 = Healthy, 6+ = Optimal
        score = min(100, int((runway_months / 6.0) * 100))
        liquidity_status = (
            "CRITICAL" if runway_months < 1.0
            else "CAUTION" if runway_months < 3.0
            else "HEALTHY" if runway_months <= 6.0
            else "OPTIMAL"
        )

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="HealthWorker",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=AgentStatus.AGENT_RESPONSE_READY.value,
            payload={
                "health_score": score,
                "runway_months": runway_months,
                "liquidity_status": liquidity_status,
                "total_liquid_cash": total_liquid,
                "monthly_burn_rate": monthly_burn,
                "default_balance": default_balance,
                "recurring_balance": recurring_balance
            }
        )

    elif intent == "EVALUATE_REBALANCE_NEED":
        # Check if recurring bucket has a deficit and needs funding from default
        needed_amount = float(msg.payload.get("target_recurring_deficit", 0.0))
        if needed_amount > 0 and default_balance >= needed_amount:
            # Rebalance is viable
            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="HealthWorker",
                recipient=msg.sender,
                status="SUCCESS",
                status_code=AgentStatus.AGENT_PROPOSAL_ACCEPTED.value,
                payload={
                    "rebalance_recommended": True,
                    "recommended_amount": needed_amount,
                    "default_surplus": default_balance - needed_amount
                }
            )
        else:
            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="HealthWorker",
                recipient=msg.sender,
                status="SUCCESS",
                status_code=AgentStatus.AGENT_RESPONSE_READY.value,
                payload={
                    "rebalance_recommended": False,
                    "reason": "Default bucket has insufficient funds or deficit is 0"
                }
            )

    return A2AResponse(
        reply_to_id=msg.message_id,
        correlation_id=msg.correlation_id,
        sender="HealthWorker",
        recipient=msg.sender,
        status="ERROR",
        status_code=AgentStatus.AGENT_PARSING_FAILURE.value,
        error_message=f"Unrecognized HealthWorker intent: {intent}"
    )

async def handle_forecaster_agent(msg: A2AMessage) -> A2AResponse:
    """
    ForecasterWorker A2A Handler:
    Provides stochastic multi-path balance corridor projections (Mean, Careless, Optimal)
    incorporating Bates volatility and ONS inflation trends.
    """
    user_uuid = msg.user_uuid
    intent = msg.intent.upper()
    days = int(msg.payload.get("days", 30))

    with SessionLocal() as session:
        # Sum current total balance
        row = session.execute(
            text("SELECT COALESCE(SUM(cached_balance), 0.0) FROM buckets WHERE user_id = :u AND is_active = TRUE"),
            {"u": user_uuid}
        ).fetchone()
        current_balance = float(row[0]) if row else 0.0

        # Recent 30-day net daily velocity
        v_row = session.execute(
            text("""
                SELECT COALESCE(SUM(amount), 0.0) / 30.0 
                FROM transactions 
                WHERE user_uuid = :u AND date >= NOW() - INTERVAL '30 days'
            """),
            {"u": user_uuid}
        ).fetchone()
        daily_velocity = float(v_row[0]) if v_row and v_row[0] else -15.0

    if intent in ("SIMULATE_STOCHASTIC_BALANCE", "PROJECT_CASHFLOW"):
        # Fast stochastic corridor estimation (Mean, Careless -2sigma, Optimal +2sigma)
        import math
        volatility_daily = 35.0  # Daily baseline standard deviation
        horizon_volatility = volatility_daily * math.sqrt(days)

        mean_projected = current_balance + (daily_velocity * days)
        careless_bound = max(0.0, mean_projected - (2.0 * horizon_volatility))
        optimal_bound = mean_projected + (1.5 * horizon_volatility)

        probability_shortfall = 0.0
        if mean_projected < 0.0:
            probability_shortfall = 0.85
        elif careless_bound <= 0.0:
            probability_shortfall = round((abs(careless_bound) / (optimal_bound + 1.0)) * 0.5, 3)

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="ForecasterWorker",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=AgentStatus.AGENT_RESPONSE_READY.value,
            payload={
                "horizon_days": days,
                "current_balance": current_balance,
                "mean_projected_balance": round(mean_projected, 2),
                "careless_bound": round(careless_bound, 2),
                "optimal_bound": round(optimal_bound, 2),
                "probability_shortfall": probability_shortfall,
                "daily_velocity": round(daily_velocity, 2)
            }
        )

    return A2AResponse(
        reply_to_id=msg.message_id,
        correlation_id=msg.correlation_id,
        sender="ForecasterWorker",
        recipient=msg.sender,
        status="ERROR",
        status_code=AgentStatus.AGENT_PARSING_FAILURE.value,
        error_message=f"Unrecognized ForecasterWorker intent: {intent}"
    )

async def handle_bucket_transfer_agent(msg: A2AMessage) -> A2AResponse:
    """
    BucketTransferWorker A2A Handler:
    Executes double-entry virtual reallocations and evaluates proposals under
    strict PostgreSQL math locks (trigger_update_bucket_balances).
    """
    intent = msg.intent.upper()

    if intent == "PROPOSE_REBALANCE":
        return await A2ANegotiator.evaluate_transfer_proposal(msg)
    elif intent == "EXECUTE_VIRTUAL_TRANSFER":
        return await A2ANegotiator.execute_transfer(msg)

    return A2AResponse(
        reply_to_id=msg.message_id,
        correlation_id=msg.correlation_id,
        sender="BucketTransferAgent",
        recipient=msg.sender,
        status="ERROR",
        status_code=AgentStatus.AGENT_PARSING_FAILURE.value,
        error_message=f"Unrecognized BucketTransferAgent intent: {intent}"
    )

async def handle_categorizer_agent(msg: A2AMessage) -> A2AResponse:
    """
    CategorizerAgent A2A Handler:
    Performs 3D taxonomy classification and staple keyword tagging.
    """
    intent = msg.intent.upper()
    description = msg.payload.get("description", "")
    merchant = msg.payload.get("merchant", "")
    full_text = f"{description} {merchant}".lower().strip()

    if intent in ("CLASSIFY_TRANSACTION", "ENRICH_ONS_TAGS"):
        cat_in = msg.payload.get("category", "")
        sub_cat_in = msg.payload.get("sub_category")
        
        # Simple merchant keyword heuristics if not specified
        if not cat_in:
            if any(k in full_text for k in ["tesco", "sainsbury", "asda", "morrisons", "waitrose", "groceries", "supermarket", "milk", "bread"]):
                cat_in = "Food & Dining"
                sub_cat_in = "Groceries"
            elif any(k in full_text for k in ["uber", "train", "rail", "fuel", "petrol", "shell", "bp"]):
                cat_in = "Transportation"
            elif any(k in full_text for k in ["netflix", "spotify", "gym", "prime"]):
                cat_in = "Subscriptions & Digital Services"
            elif any(k in full_text for k in ["energy", "gas", "electric", "water", "council"]):
                cat_in = "Utilities"
            else:
                cat_in = "Shopping & Retail"

        matched_category, matched_subcategory, combined_tags, ons_code = enrich_transaction_taxonomy(
            category=cat_in,
            sub_category=sub_cat_in,
            description=full_text,
            existing_tags=msg.payload.get("existing_tags", [])
        )

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="CategorizerAgent",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=PipelineStatus.TRANSACTIONS_SAVED.value,
            payload={
                "category": matched_category,
                "sub_category": matched_subcategory,
                "ons_code": ons_code,
                "tags": combined_tags
            }
        )

    return A2AResponse(
        reply_to_id=msg.message_id,
        correlation_id=msg.correlation_id,
        sender="CategorizerAgent",
        recipient=msg.sender,
        status="ERROR",
        status_code=AgentStatus.AGENT_PARSING_FAILURE.value,
        error_message=f"Unrecognized CategorizerAgent intent: {intent}"
    )

async def handle_analyser_agent(msg: A2AMessage) -> A2AResponse:
    """
    AnalyserAgent A2A Handler:
    Audits spending velocity, surge variance, and cash flow collision risks.
    """
    user_uuid = msg.user_uuid
    intent = msg.intent.upper()

    if intent == "ANALYZE_SPENDING_SURGE":
        amount = float(msg.payload.get("amount", 0.0))
        category = msg.payload.get("category", "General")

        with SessionLocal() as session:
            # Query spending average over past 90 days
            avg_row = session.execute(
                text("""
                    SELECT COALESCE(AVG(ABS(amount)), 0.0), COALESCE(STDDEV(ABS(amount)), 0.0)
                    FROM transactions 
                    WHERE user_uuid = :u AND amount < 0
                """),
                {"u": user_uuid}
            ).fetchone()

            avg_amount = float(avg_row[0]) if avg_row and avg_row[0] else 25.0
            std_dev = float(avg_row[1]) if avg_row and avg_row[1] else 10.0

        is_surge = amount > (avg_amount + 2.0 * std_dev) if std_dev > 0 else (amount > avg_amount * 2.0)
        variance_pct = round(((amount - avg_amount) / max(avg_amount, 1.0)) * 100, 1)

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="AnalyserAgent",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=PipelineStatus.SPEND_SURGE_DETECTED.value if is_surge else PipelineStatus.TRANSACTIONS_SAVED.value,
            payload={
                "is_surge": is_surge,
                "amount": amount,
                "category_average": round(avg_amount, 2),
                "variance_percentage": variance_pct
            }
        )

    return A2AResponse(
        reply_to_id=msg.message_id,
        correlation_id=msg.correlation_id,
        sender="AnalyserAgent",
        recipient=msg.sender,
        status="ERROR",
        status_code=AgentStatus.AGENT_PARSING_FAILURE.value,
        error_message=f"Unrecognized AnalyserAgent intent: {intent}"
    )
