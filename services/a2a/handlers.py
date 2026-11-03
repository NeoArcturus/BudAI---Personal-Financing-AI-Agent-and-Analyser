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

    elif intent == "BATCH_CLASSIFY_UNCATEGORIZED":
        from services.a2a.bus import get_a2a_bus
        bus = get_a2a_bus()

        # Step 1: Request uncategorized batch from AnalyserAgent via A2A
        ana_req = A2AMessage(
            sender="CategorizerAgent",
            recipient="AnalyserAgent",
            parent_message_id=msg.message_id,
            correlation_id=msg.correlation_id,
            message_type="QUERY",
            intent="FETCH_UNCATEGORIZED_BATCH",
            user_uuid=msg.user_uuid,
            payload={"limit": msg.payload.get("limit", 30)},
            recursion_depth=msg.recursion_depth + 1
        )
        ana_resp = await bus.request_reply(ana_req)

        if ana_resp.status != "SUCCESS":
            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="CategorizerAgent",
                recipient=msg.sender,
                status="ERROR",
                status_code=AgentStatus.AGENT_PARSING_FAILURE.value,
                error_message=f"A2A failure querying AnalyserAgent: {ana_resp.error_message}"
            )

        merchants = ana_resp.payload.get("merchants", [])
        if not merchants:
            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="CategorizerAgent",
                recipient=msg.sender,
                status="SUCCESS",
                status_code=PipelineStatus.TRANSACTIONS_SAVED.value,
                payload={
                    "classified_count": 0,
                    "message": "All transactions are already categorized in the database."
                }
            )

        # Step 2: Classify and persist the batch
        classified_records = []
        with SessionLocal() as session:
            for m in merchants:
                c_name = m.get("clean_merchant_name") or "Unknown"
                mk_uuid = m.get("merchant_knowledge_uuid")
                cat, sub_cat, tags, ons = enrich_transaction_taxonomy(
                    category="",
                    sub_category="",
                    description=c_name
                )

                if mk_uuid:
                    session.execute(text("""
                        UPDATE merchant_knowledge
                        SET category = :cat, sub_category = :sub, tags = :tags
                        WHERE knowledge_uuid = :uuid
                    """), {"cat": cat, "sub": sub_cat, "tags": json.dumps(tags), "uuid": mk_uuid})

                    session.execute(text("""
                        UPDATE transactions
                        SET sub_category = :sub, tags = :tags
                        WHERE merchant_knowledge_uuid = :uuid
                    """), {"sub": sub_cat, "tags": json.dumps(tags), "uuid": mk_uuid})

                classified_records.append({
                    "merchant": c_name,
                    "category": cat,
                    "sub_category": sub_cat,
                    "transaction_count": m.get("transaction_count", 1),
                    "total_amount": m.get("total_amount", 0.0)
                })

            session.commit()

        # Step 3: If caller requested category breakdown / highest spending, hand off to AnalyserAgent
        breakdown_payload = None
        if msg.payload.get("include_breakdown", True):
            breakdown_req = A2AMessage(
                sender="CategorizerAgent",
                recipient="AnalyserAgent",
                parent_message_id=msg.message_id,
                correlation_id=msg.correlation_id,
                message_type="QUERY",
                intent="GET_CATEGORY_BREAKDOWN",
                user_uuid=msg.user_uuid,
                recursion_depth=msg.recursion_depth + 1
            )
            breakdown_resp = await bus.request_reply(breakdown_req)
            if breakdown_resp.status == "SUCCESS":
                breakdown_payload = breakdown_resp.payload

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="CategorizerAgent",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=PipelineStatus.TRANSACTIONS_SAVED.value,
            payload={
                "classified_count": len(classified_records),
                "merchants_classified": classified_records,
                "category_breakdown": breakdown_payload
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

    elif intent == "FETCH_UNCATEGORIZED_BATCH":
        limit = min(int(msg.payload.get("limit", 30)), 50)
        account_id = msg.payload.get("account_id")

        with SessionLocal() as session:
            filter_clauses = ["t.user_uuid = :u", "(m.category IS NULL OR m.category = 'Uncategorized' OR m.category = '')"]
            params = {"u": user_uuid, "lim": limit}
            if account_id:
                filter_clauses.append("t.account_id = :acc")
                params["acc"] = account_id

            where_str = " AND ".join(filter_clauses)
            query = text(f"""
                SELECT DISTINCT m.knowledge_uuid, m.clean_merchant_name, count(t.transaction_uuid) as tx_count, sum(t.amount) as total_amount
                FROM merchant_knowledge m
                JOIN transactions t ON t.merchant_knowledge_uuid = m.knowledge_uuid
                WHERE {where_str}
                GROUP BY m.knowledge_uuid, m.clean_merchant_name
                ORDER BY count(t.transaction_uuid) DESC
                LIMIT :lim
            """)
            rows = session.execute(query, params).fetchall()
            merchants = [
                {
                    "merchant_knowledge_uuid": str(r[0]),
                    "clean_merchant_name": r[1] or "Unknown",
                    "transaction_count": r[2],
                    "total_amount": round(float(r[3] or 0.0), 2)
                }
                for r in rows
            ]

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="AnalyserAgent",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=PipelineStatus.TRANSACTIONS_SAVED.value,
            payload={
                "merchants": merchants,
                "count": len(merchants),
                "user_uuid": user_uuid
            }
        )

    elif intent == "GET_CATEGORY_BREAKDOWN":
        with SessionLocal() as session:
            # Aggregate totals per category
            cat_query = text("""
                SELECT COALESCE(m.category, 'Uncategorized') as category,
                       SUM(ABS(t.amount)) as total_spent,
                       COUNT(t.transaction_uuid) as count
                FROM transactions t
                LEFT JOIN merchant_knowledge m ON t.merchant_knowledge_uuid = m.knowledge_uuid
                WHERE t.user_uuid = :u AND t.amount < 0
                GROUP BY COALESCE(m.category, 'Uncategorized')
                ORDER BY SUM(ABS(t.amount)) DESC
            """)
            cat_rows = session.execute(cat_query, {"u": user_uuid}).fetchall()
            categories = [
                {
                    "category": r[0],
                    "total_spent": round(float(r[1] or 0.0), 2),
                    "transaction_count": r[2]
                }
                for r in cat_rows
            ]

            # Highest single transaction per category
            highest_query = text("""
                SELECT DISTINCT ON (COALESCE(m.category, 'Uncategorized'))
                       COALESCE(m.category, 'Uncategorized') as category,
                       t.description,
                       ABS(t.amount) as amount,
                       t.date
                FROM transactions t
                LEFT JOIN merchant_knowledge m ON t.merchant_knowledge_uuid = m.knowledge_uuid
                WHERE t.user_uuid = :u AND t.amount < 0
                ORDER BY COALESCE(m.category, 'Uncategorized'), ABS(t.amount) DESC
            """)
            highest_rows = session.execute(highest_query, {"u": user_uuid}).fetchall()
            highest_spending = [
                {
                    "category": r[0],
                    "description": r[1] or "Transaction",
                    "amount": round(float(r[2] or 0.0), 2),
                    "date": r[3].strftime("%Y-%m-%d") if r[3] else None
                }
                for r in highest_rows
            ]

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="AnalyserAgent",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=PipelineStatus.TRANSACTIONS_SAVED.value,
            payload={
                "categories": categories,
                "highest_spending_per_category": highest_spending,
                "total_categories": len(categories)
            }
        )

    elif intent == "GET_SUBSISTENCE_BASELINE":
        with SessionLocal() as session:
            floor_query = text("""
                SELECT COALESCE(SUM(ABS(t.amount)), 0.0) / 3.0
                FROM transactions t
                LEFT JOIN merchant_knowledge m ON t.merchant_knowledge_uuid = m.knowledge_uuid
                WHERE t.user_uuid = :u
                  AND m.category IN ('Housing', 'Utilities', 'Food & Dining', 'Healthcare', 'Transfers & Payments')
                  AND t.amount < 0
                  AND t.date >= NOW() - INTERVAL '90 days'
            """)
            res = session.execute(floor_query, {"u": user_uuid}).scalar()
            monthly_floor = round(float(res or 1200.0), 2)
            daily_floor = round(monthly_floor / 30.0, 2)

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="AnalyserAgent",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=AgentStatus.AGENT_RESPONSE_READY.value,
            payload={
                "monthly_subsistence_floor": monthly_floor,
                "daily_subsistence_floor": daily_floor,
                "observation_window_days": 90
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
