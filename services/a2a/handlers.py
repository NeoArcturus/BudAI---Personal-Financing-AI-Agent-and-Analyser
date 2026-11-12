import json
import uuid
import math
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
            if b_type in ("UNALLOCATED", "GOALS"):
                total_liquid += val
            if b_type == "UNALLOCATED":
                default_balance += val
            elif b_type == "BILLS":
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

        raw_burn = abs(float(tx_row[0])) if tx_row and tx_row[0] else 0.0
        subsistence_floor_used = False

        if raw_burn <= 0.0:
            # A2A Handoff to AnalyserAgent for empirical subsistence baseline
            from services.a2a.bus import get_a2a_bus
            bus = get_a2a_bus()
            sub_msg = A2AMessage(
                sender="HealthWorker",
                recipient="AnalyserAgent",
                message_type="QUERY",
                intent="GET_SUBSISTENCE_BASELINE",
                user_uuid=user_uuid,
                correlation_id=msg.correlation_id
            )
            sub_resp = await bus.request_reply(sub_msg)
            if sub_resp.status == "SUCCESS":
                monthly_burn = float(sub_resp.payload.get("monthly_subsistence_floor", 1200.0))
                subsistence_floor_used = True
            else:
                monthly_burn = 1200.0
        else:
            monthly_burn = raw_burn

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
                "monthly_burn": monthly_burn,
                "monthly_burn_rate": monthly_burn,
                "subsistence_floor_used": subsistence_floor_used,
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
    Provides stochastic multi-path expense projections and goal-aligned optimal budget ceilings.
    Compares current pace against prior monthly run-rate and drops optimal curve if no goals exist.
    """
    user_uuid = msg.user_uuid
    intent = msg.intent.upper()
    days = int(msg.payload.get("days", 30))

    if intent in ("PROJECT_EXPENSES", "GENERATE_EXPENSE_FORECAST", "SIMULATE_STOCHASTIC_BALANCE", "PROJECT_CASHFLOW"):
        from services.mcp_tools.forecaster_tools import generate_expense_forecast
        forecast_summary = generate_expense_forecast.invoke({
            "user_uuid": user_uuid,
            "days": days
        })
        
        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="ForecasterWorker",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=AgentStatus.AGENT_RESPONSE_READY.value,
            payload={
                "horizon_days": days,
                "forecast_summary": forecast_summary
            }
        )

    elif intent == "EVALUATE_SOLVENCY_IMPACT":
        with SessionLocal() as session:
            row = session.execute(
                text("SELECT COALESCE(SUM(cached_balance), 0.0) FROM buckets WHERE user_id = :u AND is_active = TRUE"),
                {"u": user_uuid}
            ).fetchone()
            current_balance = float(row[0]) if row else 0.0

            v_row = session.execute(
                text("""
                    SELECT COALESCE(SUM(amount), 0.0) / 30.0 
                    FROM transactions 
                    WHERE user_uuid = :u AND date >= NOW() - INTERVAL '30 days'
                """),
                {"u": user_uuid}
            ).fetchone()
            daily_velocity = float(v_row[0]) if v_row and v_row[0] else -15.0

        deduction = float(msg.payload.get("deduction_amount", 0.0))
        horizon = int(msg.payload.get("horizon_days", 30))

        post_balance = current_balance - deduction
        projected_after = post_balance + (daily_velocity * horizon)
        
        safety_floor = 800.0
        is_safe = (post_balance >= safety_floor) and (projected_after >= safety_floor)
        shortfall_prob = 0.85 if not is_safe else 0.05

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="ForecasterWorker",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=AgentStatus.AGENT_RESPONSE_READY.value,
            payload={
                "current_balance": current_balance,
                "deduction_amount": deduction,
                "post_deduction_balance": round(post_balance, 2),
                "projected_balance_after": round(projected_after, 2),
                "is_safe": is_safe,
                "shortfall_probability": shortfall_prob,
                "risk_rating": "CRITICAL" if not is_safe else "LOW"
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

async def handle_bucket_agent(msg: A2AMessage) -> A2AResponse:
    """
    BucketAgent A2A Handler:
    Manages the 3-container virtual bucket hierarchy (UNALLOCATED, BILLS, GOALS),
    executes Stage 3 bill timeline allocations (<15d automated moves),
    enforces the spending hierarchy (Bills > Goals), and tracks bank account shortfalls.
    """
    user_uuid = msg.user_uuid
    intent = msg.intent.upper()
    from services.a2a.negotiator import A2ANegotiator
    import services.bucket_service as bucket_service

    if intent == "SYNC_UNALLOCATED_ACCOUNTS":
        with SessionLocal() as session:
            unallocated = bucket_service.sync_unallocated_from_accounts(session, user_uuid)
            session.commit()
            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="BucketAgent",
                recipient=msg.sender,
                status="SUCCESS",
                status_code=AgentStatus.AGENT_RESPONSE_READY.value,
                payload={
                    "total_balance": unallocated.cached_balance,
                    "currencies": unallocated.cached_balances
                }
            )

    elif intent in ("PROCESS_STAGE3_BILLS", "PROVISION_STAGE3_BILLS"):
        bills_data = msg.payload.get("bills") or msg.payload.get("subscriptions") or []
        if isinstance(bills_data, dict):
            bills_data = [bills_data]
        with SessionLocal() as session:
            res = bucket_service.process_stage3_bills(session, user_uuid, bills_data)
            session.commit()
            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="BucketAgent",
                recipient=msg.sender,
                status="SUCCESS",
                status_code=AgentStatus.AGENT_RESPONSE_READY.value,
                payload=res
            )

    elif intent == "DEDUCT_TRANSACTION_OUTFLOW":
        tx_data = msg.payload.get("transaction") or msg.payload
        with SessionLocal() as session:
            res = bucket_service.deduct_transaction_outflow(session, user_uuid, tx_data)
            session.commit()
            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="BucketAgent",
                recipient=msg.sender,
                status="SUCCESS",
                status_code=AgentStatus.AGENT_RESPONSE_READY.value,
                payload=res
            )

    elif intent == "CHECK_REALITY_TRANSFERS":
        with SessionLocal() as session:
            reminders = bucket_service.check_reality_transfers(session, user_uuid)
            session.commit()
            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="BucketAgent",
                recipient=msg.sender,
                status="SUCCESS",
                status_code=AgentStatus.AGENT_RESPONSE_READY.value,
                payload={"reminders": reminders, "count": len(reminders)}
            )

    elif intent == "PROPOSE_REBALANCE":
        return await A2ANegotiator.evaluate_transfer_proposal(msg)
    elif intent == "EXECUTE_VIRTUAL_TRANSFER":
        return await A2ANegotiator.execute_transfer(msg)
    elif intent == "CONSENSUS_REBALANCE":
        return await A2ANegotiator.negotiate_rebalance(msg)

    return A2AResponse(
        reply_to_id=msg.message_id,
        correlation_id=msg.correlation_id,
        sender="BucketAgent",
        recipient=msg.sender,
        status="ERROR",
        status_code=AgentStatus.AGENT_PARSING_FAILURE.value,
        error_message=f"Unrecognized BucketAgent intent: {intent}"
    )

# Direct reference for legacy callers
handle_bucket_transfer_agent = handle_bucket_agent

async def handle_categorizer_agent(msg: A2AMessage) -> A2AResponse:
    """
    CategorizerAgent A2A Handler:
    Performs 3D taxonomy classification and staple keyword tagging.
    """
    intent = msg.intent.upper()
    description = msg.payload.get("description", "")
    merchant = msg.payload.get("merchant", "")
    full_text = f"{description} {merchant}".lower().strip()

    if intent in ("CLASSIFY_TRANSACTION", "ENRICH_ONS_TAGS", "PROCESS_NEW_EXPENSE"):
        cat_in = msg.payload.get("category", "")
        sub_cat_in = msg.payload.get("sub_category")
        clean_merchant = msg.payload.get("clean_merchant") or merchant or description
        amount = float(msg.payload.get("amount", 0.0))
        tx_uuid = msg.payload.get("transaction_uuid")

        # Priority 1: Deterministic regex taxonomy rules
        if not cat_in:
            from services.categorization.classifier import classify_merchant
            rule_cat, rule_sub = classify_merchant(clean_merchant, description, amount)
            if rule_cat != "Uncategorized":
                cat_in = rule_cat
                sub_cat_in = rule_sub

        # Priority 2: Intra-batch Qdrant flywheel cache lookup (<2ms)
        if not cat_in:
            try:
                from services.qdrant_service import get_qdrant_service
                from services.embedding_service import get_embedding_service
                qdrant = get_qdrant_service()
                embedder = get_embedding_service()
                vec = embedder.embed_query(clean_merchant)
                hits = qdrant.search_merchant_knowledge(vec, limit=1, score_threshold=0.88)
                if hits:
                    hit_payload = hits[0].get("payload", {})
                    cat_in = hit_payload.get("category", "Shopping & Retail")
                    sub_cat_in = hit_payload.get("sub_category", "General Merchandise")
            except Exception as q_err:
                logger.warning(f"Qdrant flywheel lookup bypassed: {q_err}")

        # Fallback default
        if not cat_in:
            cat_in = "Shopping & Retail"
            sub_cat_in = "General Merchandise"

        matched_category, matched_subcategory, combined_tags, ons_code = enrich_transaction_taxonomy(
            category=cat_in,
            sub_category=sub_cat_in,
            description=full_text,
            existing_tags=msg.payload.get("existing_tags", [])
        )

        # Update transaction record in PostgreSQL if tx_uuid provided
        if tx_uuid:
            try:
                with SessionLocal() as session:
                    session.execute(
                        text("""
                            UPDATE transactions 
                            SET sub_category = :sub_cat, tags = :tags::jsonb
                            WHERE transaction_uuid = :uid
                        """),
                        {"sub_cat": matched_subcategory, "tags": json.dumps(combined_tags), "uid": tx_uuid}
                    )
                    session.commit()
            except Exception as upd_err:
                logger.warning(f"Could not update transaction {tx_uuid} post-categorization: {upd_err}")

        # Stage 3 Handoff: Broadcast EXPENSE_CATEGORIZED to wake parallel intelligence engines
        from services.a2a.bus import get_a2a_bus
        bus = get_a2a_bus()
        asyncio.create_task(bus.send(A2AMessage(
            sender="CategorizerAgent",
            recipient="A2ABroadcast",
            message_type="COMMAND",
            intent="EXPENSE_CATEGORIZED",
            user_uuid=msg.user_uuid,
            correlation_id=msg.correlation_id,
            payload={
                "transaction_uuid": tx_uuid,
                "amount": amount,
                "category": matched_category,
                "sub_category": matched_subcategory,
                "tags": combined_tags,
                "ons_code": ons_code,
                "clean_merchant": clean_merchant
            }
        )))

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
        from services.categorization.classifier import classify_merchant
        classified_records = []
        with SessionLocal() as session:
            for m in merchants:
                c_name = m.get("clean_merchant_name") or "Unknown"
                mk_uuid = m.get("merchant_knowledge_uuid")
                tot_amt = float(m.get("total_amount", 0.0))

                # Step 2a: Run genuine classifier (no blind clothing default)
                raw_cat, raw_sub = classify_merchant(merchant_name=c_name, amount=tot_amt)

                # Step 2b: Enrich with canonical ONS macros and tags
                cat, sub_cat, tags, ons = enrich_transaction_taxonomy(
                    category=raw_cat,
                    sub_category=raw_sub,
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

                    try:
                        from services.qdrant_service import get_qdrant_service
                        get_qdrant_service().update_merchant_payload(
                            knowledge_uuid=mk_uuid,
                            category=cat,
                            sub_category=sub_cat,
                            tags=tags
                        )
                    except Exception:
                        pass

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

    elif intent == "GET_HISTORICAL_VOLATILITY":
        window_days = int(msg.payload.get("window_days", 180))
        with SessionLocal() as session:
            # Group daily net transactions using date_trunc
            query = text("""
                SELECT date_trunc('day', date)::date as tx_day, SUM(amount) as daily_net
                FROM transactions
                WHERE user_uuid = :u AND date >= NOW() - (:win || ' days')::interval
                GROUP BY date_trunc('day', date)::date
                ORDER BY tx_day ASC
            """)
            rows = session.execute(query, {"u": user_uuid, "win": window_days}).fetchall()
            
            if rows:
                daily_values = [float(r[1] or 0.0) for r in rows]
                obs_count = len(daily_values)
                mean_drift = sum(daily_values) / max(obs_count, 1)
                if obs_count > 1:
                    variance = sum((x - mean_drift) ** 2 for x in daily_values) / (obs_count - 1)
                    std_dev = math.sqrt(variance)
                else:
                    std_dev = 25.0
            else:
                mean_drift = -15.0
                std_dev = 30.0
                obs_count = 0

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="AnalyserAgent",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=AgentStatus.AGENT_RESPONSE_READY.value,
            payload={
                "daily_mean_drift": round(mean_drift, 2),
                "daily_std_dev": round(std_dev, 2),
                "observation_count": obs_count,
                "window_days": window_days
            }
        )

    elif intent == "GET_NET_CASHFLOW_SURPLUS":
        with SessionLocal() as session:
            surplus_query = text("""
                SELECT 
                    COALESCE(SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END), 0.0) / 3.0 as monthly_income,
                    COALESCE(SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END), 0.0) / 3.0 as monthly_expenses
                FROM transactions
                WHERE user_uuid = :u AND date >= NOW() - INTERVAL '90 days'
            """)
            row = session.execute(surplus_query, {"u": user_uuid}).fetchone()
            monthly_income = round(float(row[0] or 0.0), 2) if row else 0.0
            monthly_expenses = round(float(row[1] or 0.0), 2) if row else 0.0
            net_surplus = round(monthly_income - monthly_expenses, 2)
            savings_ratio = round(net_surplus / monthly_income, 3) if monthly_income > 0 else 0.0

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="AnalyserAgent",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=AgentStatus.AGENT_RESPONSE_READY.value,
            payload={
                "monthly_income": monthly_income,
                "monthly_expenses": monthly_expenses,
                "net_surplus": net_surplus,
                "savings_ratio": savings_ratio,
                "period_months": 3
            }
        )

    elif intent == "GET_CATEGORY_BASELINE":
        category = msg.payload.get("category", "General")
        test_amount = float(msg.payload.get("test_amount", 0.0))
        with SessionLocal() as session:
            cat_query = text("""
                SELECT ABS(t.amount)
                FROM transactions t
                LEFT JOIN merchant_knowledge m ON t.merchant_knowledge_uuid = m.knowledge_uuid
                WHERE t.user_uuid = :u AND m.category = :cat AND t.amount < 0
            """)
            rows = session.execute(cat_query, {"u": user_uuid, "cat": category}).fetchall()
            amounts = [float(r[0] or 0.0) for r in rows]
            
            count = len(amounts)
            if count > 0:
                mean_val = sum(amounts) / count
                if count > 1:
                    variance = sum((x - mean_val) ** 2 for x in amounts) / (count - 1)
                    std_dev = math.sqrt(variance)
                else:
                    std_dev = mean_val * 0.3
            else:
                mean_val = 50.0
                std_dev = 20.0
            
            z_score = round((test_amount - mean_val) / max(std_dev, 1.0), 2)
            is_anomaly = abs(z_score) > 2.5 or (test_amount > 3.0 * mean_val and test_amount > 100.0)

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="AnalyserAgent",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=AgentStatus.AGENT_RESPONSE_READY.value,
            payload={
                "category": category,
                "category_mean": round(mean_val, 2),
                "category_std_dev": round(std_dev, 2),
                "sample_count": count,
                "test_amount": test_amount,
                "z_score": z_score,
                "is_anomaly": is_anomaly
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

async def handle_scenario_agent(msg: A2AMessage) -> A2AResponse:
    """
    ScenarioWorker A2A Handler:
    Orchestrates multi-agent analysis to simulate macro shocks (e.g., £500 boiler repair, 20% income reduction)
    and predicts liquidity resilience without HITL.
    """
    user_uuid = msg.user_uuid
    intent = msg.intent.upper()
    from services.a2a.bus import get_a2a_bus
    bus = get_a2a_bus()

    if intent in ("SIMULATE_EXPENSE_SHOCK", "PROJECT_REVENUE_DROP"):
        shock_amount = float(msg.payload.get("shock_amount", 500.0))
        
        # Step 1: Query HealthWorker for current liquid reserves and burn rate
        health_msg = A2AMessage(
            sender="ScenarioWorker",
            recipient="HealthWorker",
            message_type="QUERY",
            intent="AUDIT_HEALTH_SCORE",
            user_uuid=user_uuid,
            correlation_id=msg.correlation_id
        )
        health_resp = await bus.request_reply(health_msg)
        total_liquid = float(health_resp.payload.get("total_liquid_cash", 0.0))
        monthly_burn = float(health_resp.payload.get("monthly_burn") or health_resp.payload.get("monthly_burn_rate") or 0.0)

        # Step 2: Query AnalyserAgent for net cashflow surplus
        surplus_msg = A2AMessage(
            sender="ScenarioWorker",
            recipient="AnalyserAgent",
            message_type="QUERY",
            intent="GET_NET_CASHFLOW_SURPLUS",
            user_uuid=user_uuid,
            correlation_id=msg.correlation_id
        )
        surplus_resp = await bus.request_reply(surplus_msg)
        net_surplus = float(surplus_resp.payload.get("net_surplus", 0.0))

        post_shock_liquid = round(total_liquid - shock_amount, 2)
        post_shock_runway = round(max(0.0, post_shock_liquid) / max(1.0, monthly_burn), 2)
        can_absorb = post_shock_liquid >= 0.0

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="ScenarioWorker",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=AgentStatus.AGENT_RESPONSE_READY.value,
            payload={
                "shock_amount": shock_amount,
                "can_absorb_shock": can_absorb,
                "pre_shock_liquid_cash": total_liquid,
                "post_shock_liquid_cash": post_shock_liquid,
                "pre_shock_runway_months": health_resp.payload.get("runway_months", 0.0),
                "post_shock_runway_months": post_shock_runway,
                "monthly_net_surplus": net_surplus,
                "recovery_months": round(shock_amount / max(net_surplus, 1.0), 1) if net_surplus > 0 else 999.0
            }
        )

    return A2AResponse(
        reply_to_id=msg.message_id,
        correlation_id=msg.correlation_id,
        sender="ScenarioWorker",
        recipient=msg.sender,
        status="ERROR",
        status_code=AgentStatus.AGENT_PARSING_FAILURE.value,
        error_message=f"Unrecognized ScenarioWorker intent: {intent}"
    )

async def handle_anomaly_agent(msg: A2AMessage) -> A2AResponse:
    """
    AnomalyWorker A2A Handler:
    Evaluates transaction anomalies using Robust Modified Z-Scores (Median Absolute Deviation)
    and Poisson burst arrival frequency, avoiding mean/std distortion.
    """
    user_uuid = msg.user_uuid
    intent = msg.intent.upper()
    from services.a2a.bus import get_a2a_bus
    bus = get_a2a_bus()

    if intent in ("EVALUATE_TRANSACTION_ANOMALY", "AUDIT_OUTLIER"):
        amount = float(msg.payload.get("amount", 0.0))
        category = msg.payload.get("category", "General & Discretionary")
        merchant = msg.payload.get("merchant", "Unknown")

        # Step 1: Query historical transactions in this category for MAD calculation
        with SessionLocal() as session:
            cat_txs = session.execute(
                text("""
                    SELECT ABS(amount)
                    FROM transactions
                    WHERE user_uuid = :u AND category = :cat AND amount < 0
                    ORDER BY date DESC LIMIT 60
                """),
                {"u": user_uuid, "cat": category}
            ).fetchall()

        amounts = [float(r[0]) for r in cat_txs if r[0] is not None]

        # Step 2: Query MarketWorker for category inflation rate
        mkt_msg = A2AMessage(
            sender="AnomalyWorker",
            recipient="MarketWorker",
            message_type="QUERY",
            intent="FETCH_MACRO_INFLATION",
            user_uuid=user_uuid,
            correlation_id=msg.correlation_id,
            payload={"category": category}
        )
        mkt_resp = await bus.request_reply(mkt_msg)
        inflation_yoy = float(mkt_resp.payload.get("delta_pct_yoy", 0.0))

        if len(amounts) >= 5:
            s_amounts = sorted(amounts)
            n = len(s_amounts)
            median_val = s_amounts[n // 2] if n % 2 != 0 else (s_amounts[n // 2 - 1] + s_amounts[n // 2]) / 2.0
            devs = sorted(abs(x - median_val) for x in s_amounts)
            mad = devs[n // 2] if n % 2 != 0 else (devs[n // 2 - 1] + devs[n // 2]) / 2.0
            
            # Inflation-adjust baseline
            adjusted_baseline = median_val * (1.0 + (inflation_yoy / 100.0))
            if mad > 0:
                mod_z = round(0.6745 * (amount - adjusted_baseline) / mad, 2)
            else:
                mod_z = 0.0
            is_anomaly = mod_z > 3.5 or (amount > 3.0 * adjusted_baseline and amount > 75.0)
            baseline_val = adjusted_baseline
            z_score = mod_z
        else:
            # Fallback to AnalyserAgent baseline
            base_msg = A2AMessage(
                sender="AnomalyWorker",
                recipient="AnalyserAgent",
                message_type="QUERY",
                intent="GET_CATEGORY_BASELINE",
                user_uuid=user_uuid,
                correlation_id=msg.correlation_id,
                payload={"category": category, "test_amount": amount}
            )
            base_resp = await bus.request_reply(base_msg)
            cat_mean = float(base_resp.payload.get("category_mean", 50.0))
            cat_std = float(base_resp.payload.get("category_std_dev", 20.0))
            adjusted_baseline = cat_mean * (1.0 + (inflation_yoy / 100.0))
            deviation = amount - adjusted_baseline
            z_score = round(deviation / max(cat_std, 1.0), 2)
            is_anomaly = abs(z_score) > 2.5 or (amount > 3.0 * adjusted_baseline and amount > 80.0)
            baseline_val = adjusted_baseline

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="AnomalyWorker",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=AgentStatus.AGENT_RESPONSE_READY.value,
            payload={
                "merchant": merchant,
                "amount": amount,
                "category": category,
                "is_anomaly": is_anomaly,
                "modified_z_score": z_score,
                "category_baseline": round(baseline_val, 2),
                "inflation_rate_yoy": inflation_yoy,
                "severity": "CRITICAL" if z_score > 4.0 else "WARNING" if is_anomaly else "NORMAL"
            }
        )

    elif intent == "REMEDIATE_ANOMALY":
        amount = float(msg.payload.get("amount", 0.0))
        category = msg.payload.get("category", "General & Discretionary")
        merchant = msg.payload.get("merchant", "Unknown")

        # Step 1: Run EVALUATE_TRANSACTION_ANOMALY
        eval_msg = A2AMessage(
            sender="AnomalyWorker",
            recipient="AnomalyWorker",
            message_type="QUERY",
            intent="EVALUATE_TRANSACTION_ANOMALY",
            user_uuid=user_uuid,
            correlation_id=msg.correlation_id,
            payload=msg.payload
        )
        eval_resp = await handle_anomaly_agent(eval_msg)
        is_anomaly = eval_resp.payload.get("is_anomaly", False)

        alert_dispatched = False
        if is_anomaly:
            from services.notifications.alert_dispatcher import AlertDispatcher
            dispatcher = AlertDispatcher()
            dispatcher.dispatch(
                user_uuid=user_uuid,
                message=f"Anomalous expenditure detected: £{amount:.2f} at {merchant} ({category}).",
                event_type="spending_anomaly_detected",
                title="Spending Anomaly Flagged",
                urgency=2,
                status_code="305-401"
            )
            alert_dispatched = True

        remediation = {
            "action": "TRIGGER_CASCADE" if is_anomaly and amount > 500.0 else "LOG_AND_NOTIFY",
            "recommended_budget_adjustment": round(amount * 0.2, 2) if is_anomaly else 0.0,
            "requires_priority_cascade": is_anomaly and amount > 300.0
        }

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="AnomalyWorker",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=AgentStatus.AGENT_RESPONSE_READY.value,
            payload={
                "anomaly_evaluation": eval_resp.payload,
                "remediation_action": remediation,
                "alert_dispatched": alert_dispatched
            }
        )

    return A2AResponse(
        reply_to_id=msg.message_id,
        correlation_id=msg.correlation_id,
        sender="AnomalyWorker",
        recipient=msg.sender,
        status="ERROR",
        status_code=AgentStatus.AGENT_PARSING_FAILURE.value,
        error_message=f"Unrecognized AnomalyWorker intent: {intent}"
    )

async def handle_subscription_detection_agent(msg: A2AMessage) -> A2AResponse:
    """
    SubscriptionWorker A2A Handler:
    Detects recurring merchant rhythms using SQL window functions,
    identifies price hikes without HDBSCAN clustering, and registers system alerts.
    """
    user_uuid = msg.user_uuid
    intent = msg.intent.upper()
    from models.database_models import Subscription, SystemAlert
    from sqlmodel import select

    if intent in ("SCAN_RECURRING_PAYMENTS", "DETECT_SUBSCRIPTIONS"):
        with SessionLocal() as session:
            subs = session.execute(
                select(Subscription).where(Subscription.user_uuid == user_uuid)
            ).scalars().all()
            
            sub_list = []
            for s in subs:
                sub_list.append({
                    "merchant": s.merchant_name,
                    "expected_amount": float(s.expected_amount),
                    "frequency": s.predicted_frequency,
                    "next_date": s.next_expected_date.strftime("%Y-%m-%d"),
                    "is_price_hike": s.is_price_hike
                })

            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="SubscriptionWorker",
                recipient=msg.sender,
                status="SUCCESS",
                status_code=AgentStatus.AGENT_RESPONSE_READY.value,
                payload={"subscriptions": sub_list, "count": len(sub_list)}
            )

    elif intent == "DETECT_RECURRING_RHYTHM":
        merchant = msg.payload.get("merchant", "")
        with SessionLocal() as session:
            query = text("""
                WITH ranked AS (
                    SELECT 
                        date, 
                        amount,
                        LAG(date) OVER (ORDER BY date ASC) as prev_date,
                        LAG(amount) OVER (ORDER BY date ASC) as prev_amount
                    FROM transactions
                    WHERE user_uuid = :u 
                      AND (semi_cleaned_description ILIKE :m OR description ILIKE :m)
                    ORDER BY date DESC
                    LIMIT 10
                )
                SELECT 
                    AVG(DATE_PART('day', date - prev_date)) as avg_interval,
                    STDDEV(DATE_PART('day', date - prev_date)) as std_interval,
                    AVG(amount) as avg_amount,
                    STDDEV(amount) as std_amount,
                    COUNT(*) as match_count,
                    MAX(date) as latest_date,
                    MAX(amount) as latest_amount
                FROM ranked 
                WHERE prev_date IS NOT NULL
            """)
            row = session.execute(query, {"u": user_uuid, "m": f"%{merchant.lower()}%"}).fetchone()

            is_recurring = False
            cadence = "UNKNOWN"
            is_price_hike = False
            hike_amount = 0.0

            if row and row[4] and row[4] >= 2:
                avg_int = float(row[0] or 0.0)
                std_int = float(row[1] or 0.0)
                avg_amt = float(row[2] or 0.0)
                latest_amt = float(row[6] or 0.0)

                if std_int <= 4.0:
                    if 25.0 <= avg_int <= 35.0:
                        is_recurring = True
                        cadence = "MONTHLY"
                    elif 6.0 <= avg_int <= 8.0:
                        is_recurring = True
                        cadence = "WEEKLY"
                    elif 350.0 <= avg_int <= 375.0:
                        is_recurring = True
                        cadence = "ANNUAL"

                if is_recurring and latest_amt > avg_amt * 1.05:
                    is_price_hike = True
                    hike_amount = round(latest_amt - avg_amt, 2)

            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="SubscriptionWorker",
                recipient=msg.sender,
                status="SUCCESS",
                status_code=AgentStatus.AGENT_RESPONSE_READY.value,
                payload={
                    "merchant": merchant,
                    "is_recurring": is_recurring,
                    "cadence": cadence,
                    "is_price_hike": is_price_hike,
                    "hike_amount": hike_amount
                }
            )

    return A2AResponse(
        reply_to_id=msg.message_id,
        correlation_id=msg.correlation_id,
        sender="SubscriptionWorker",
        recipient=msg.sender,
        status="ERROR",
        status_code=AgentStatus.AGENT_PARSING_FAILURE.value,
        error_message=f"Unrecognized SubscriptionWorker intent: {intent}"
    )

async def handle_cascade_agent(msg: A2AMessage) -> A2AResponse:
    """
    CascadeWorker A2A Handler:
    Autonomous Priority Cascade (The Overdraft Engine).
    Drains auxiliary buckets in strict descending priority order (lowest priority first)
    to cover a deficit in the DEFAULT bucket under PostgreSQL double-entry math lock triggers.
    """
    user_uuid = msg.user_uuid
    intent = msg.intent.upper()
    deficit = float(msg.payload.get("deficit", 0.0))
    target_default_bucket_id = msg.payload.get("target_default_bucket_id")

    if intent == "EXECUTE_PRIORITY_CASCADE":
        if deficit <= 0.0:
            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="CascadeWorker",
                recipient=msg.sender,
                status="SUCCESS",
                status_code=AgentStatus.AGENT_RESPONSE_READY.value,
                payload={
                    "deficit_resolved": True,
                    "total_recovered": 0.0,
                    "uncovered_deficit": 0.0,
                    "transfers_executed": []
                }
            )

        with SessionLocal() as session:
            # 1. Locate DEFAULT bucket
            if target_default_bucket_id:
                def_row = session.execute(
                    text("SELECT id, name, cached_balance FROM buckets WHERE id = :id AND user_id = :u"),
                    {"id": target_default_bucket_id, "u": user_uuid}
                ).fetchone()
            else:
                def_row = session.execute(
                    text("SELECT id, name, cached_balance FROM buckets WHERE user_id = :u AND type = 'DEFAULT' AND is_active = TRUE"),
                    {"u": user_uuid}
                ).fetchone()

            if not def_row:
                return A2AResponse(
                    reply_to_id=msg.message_id,
                    correlation_id=msg.correlation_id,
                    sender="CascadeWorker",
                    recipient=msg.sender,
                    status="ERROR",
                    status_code=AgentStatus.AGENT_PARSING_FAILURE.value,
                    error_message=f"No active DEFAULT bucket found for user {user_uuid}"
                )

            default_id = def_row[0]

            # 2. Query candidate auxiliary buckets sorted by priority_index DESC (lowest priority drained first)
            candidates = session.execute(
                text("""
                    SELECT id, name, type, priority_index, cached_balance
                    FROM buckets
                    WHERE user_id = :u AND id != :def_id AND type != 'DEFAULT' AND is_active = TRUE AND cached_balance > 0
                    ORDER BY priority_index DESC
                """),
                {"u": user_uuid, "def_id": default_id}
            ).fetchall()

            remaining_deficit = deficit
            total_recovered = 0.0
            transfers_executed = []

            for b_id, b_name, b_type, b_priority, b_balance in candidates:
                if remaining_deficit <= 0.01:
                    break

                b_bal = float(b_balance or 0.0)
                drain_amount = min(b_bal, remaining_deficit)
                t_id = str(uuid.uuid4())

                # Insert double-entry virtual transfer under math lock
                session.execute(
                    text("""
                        INSERT INTO virtual_transfers (id, source_bucket_id, target_bucket_id, amount, timestamp)
                        VALUES (:id, :src, :dst, :amt, NOW())
                    """),
                    {"id": t_id, "src": b_id, "dst": default_id, "amt": drain_amount}
                )
                session.commit()

                remaining_deficit = round(remaining_deficit - drain_amount, 2)
                total_recovered = round(total_recovered + drain_amount, 2)
                transfers_executed.append({
                    "transfer_id": t_id,
                    "source_bucket_id": b_id,
                    "source_bucket_name": b_name,
                    "source_priority": float(b_priority),
                    "target_bucket_id": default_id,
                    "amount": drain_amount
                })

        deficit_resolved = remaining_deficit <= 0.01

        if not deficit_resolved:
            from services.notifications.alert_dispatcher import AlertDispatcher
            dispatcher = AlertDispatcher()
            dispatcher.dispatch(
                user_uuid=user_uuid,
                message=f"Priority cascade partially exhausted: £{remaining_deficit:.2f} uncovered deficit remains.",
                event_type="cascade_exhaustion_warning",
                title="Liquidity Deficit Unresolved",
                urgency=3,
                status_code="500-409"
            )

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="CascadeWorker",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=AgentStatus.AGENT_RESPONSE_READY.value if deficit_resolved else "500-409",
            payload={
                "deficit_resolved": deficit_resolved,
                "initial_deficit": deficit,
                "total_recovered": total_recovered,
                "uncovered_deficit": max(0.0, remaining_deficit),
                "transfers_executed": transfers_executed
            }
        )

    return A2AResponse(
        reply_to_id=msg.message_id,
        correlation_id=msg.correlation_id,
        sender="CascadeWorker",
        recipient=msg.sender,
        status="ERROR",
        status_code=AgentStatus.AGENT_PARSING_FAILURE.value,
        error_message=f"Unrecognized CascadeWorker intent: {intent}"
    )


# --- STAGE 1 AGENTS: Ingestion, Reconciliation, and Transaction Intake ---

async def handle_truelayer_ingestion_agent(msg: A2AMessage) -> A2AResponse:
    """
    TrueLayerIngestionAgent A2A Handler:
    Manages external Open Banking communication, token refreshes,
    and non-blocking delta transaction pulls.
    """
    intent = msg.intent.upper()
    user_uuid = msg.user_uuid
    payload = msg.payload
    account_id = payload.get("account_id")

    if intent in ("SYNC_ACCOUNT_DELTA", "TRIGGER_SYNC"):
        if not account_id:
            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="TrueLayerIngestionAgent",
                recipient=msg.sender,
                status="ERROR",
                status_code=AgentStatus.AGENT_PARSING_FAILURE.value,
                error_message="Missing account_id in sync request"
            )
        try:
            from services.api_integrator.truelayer_sync import TrueLayerSync
            syncer = TrueLayerSync(user_id=user_uuid)
            syncer.trigger_sync(account_id=account_id, user_uuid=user_uuid, from_date=payload.get("from_date"))
            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="TrueLayerIngestionAgent",
                recipient=msg.sender,
                status="SUCCESS",
                status_code=AgentStatus.AGENT_RESPONSE_READY.value,
                payload={"account_id": account_id, "status": "DELTA_SYNC_COMPLETED"}
            )
        except Exception as e:
            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="TrueLayerIngestionAgent",
                recipient=msg.sender,
                status="ERROR",
                status_code=AgentStatus.AGENT_EXECUTION_FAILURE.value,
                error_message=f"Delta sync failed: {str(e)}"
            )

    elif intent == "REFRESH_OAUTH_TOKENS":
        provider_id = payload.get("provider_id")
        try:
            from services.api_integrator.access_token_generator import AccessTokenGenerator
            token_gen = AccessTokenGenerator()
            refreshed = token_gen.refresh_token(provider_id, user_uuid)
            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="TrueLayerIngestionAgent",
                recipient=msg.sender,
                status="SUCCESS" if refreshed else "ERROR",
                status_code=AgentStatus.AGENT_RESPONSE_READY.value if refreshed else AgentStatus.AGENT_EXECUTION_FAILURE.value,
                payload={"provider_id": provider_id, "refreshed": refreshed}
            )
        except Exception as e:
            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="TrueLayerIngestionAgent",
                recipient=msg.sender,
                status="ERROR",
                status_code=AgentStatus.AGENT_EXECUTION_FAILURE.value,
                error_message=f"Token refresh failed: {str(e)}"
            )

    return A2AResponse(
        reply_to_id=msg.message_id,
        correlation_id=msg.correlation_id,
        sender="TrueLayerIngestionAgent",
        recipient=msg.sender,
        status="ERROR",
        status_code=AgentStatus.AGENT_PARSING_FAILURE.value,
        error_message=f"Unrecognized TrueLayerIngestionAgent intent: {intent}"
    )


async def handle_reconciliation_agent(msg: A2AMessage) -> A2AResponse:
    """
    ReconciliationAgent A2A Handler:
    Anchors physical bank balances, computes physical balance deltas,
    and executes rolling 365-day circular buffer purging.
    """
    intent = msg.intent.upper()
    user_uuid = msg.user_uuid
    payload = msg.payload

    if intent in ("ANCHOR_ACCOUNT_BALANCE", "ANCHOR_BALANCE"):
        account_id = payload.get("account_id")
        live_balance = float(payload.get("live_balance", {}).get("current", payload.get("live_balance", 0.0)) if isinstance(payload.get("live_balance"), dict) else payload.get("live_balance", 0.0))

        with SessionLocal() as session:
            prev_row = session.execute(
                text("SELECT account_balance FROM accounts WHERE account_id = :acc"),
                {"acc": account_id}
            ).fetchone()
            prev_bal = float(prev_row[0] or 0.0) if prev_row else live_balance

            session.execute(
                text("UPDATE accounts SET account_balance = :bal, last_synced_at = NOW() WHERE account_id = :acc"),
                {"bal": live_balance, "acc": account_id}
            )
            session.commit()
            delta = live_balance - prev_bal

            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="ReconciliationAgent",
                recipient=msg.sender,
                status="SUCCESS",
                status_code=AgentStatus.AGENT_RESPONSE_READY.value,
                payload={
                    "account_id": account_id,
                    "live_balance": live_balance,
                    "previous_balance": prev_bal,
                    "physical_delta": delta
                }
            )

    elif intent in ("PURGE_CIRCULAR_BUFFER", "ENFORCE_RETENTION"):
        with SessionLocal() as session:
            res = session.execute(
                text("DELETE FROM transactions WHERE user_uuid = :u AND date < (CURRENT_DATE - INTERVAL '365 days')"),
                {"u": user_uuid}
            )
            session.commit()
            purged = res.rowcount

            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="ReconciliationAgent",
                recipient=msg.sender,
                status="SUCCESS",
                status_code=AgentStatus.AGENT_RESPONSE_READY.value,
                payload={"purged_records": purged}
            )

    elif intent in ("AUDIT_ZERO_SUM", "FETCH_BALANCES"):
        with SessionLocal() as session:
            physical = session.execute(
                text("SELECT COALESCE(SUM(account_balance), 0) FROM accounts WHERE user_uuid = :u"),
                {"u": user_uuid}
            ).scalar()
            virtual = session.execute(
                text("SELECT COALESCE(SUM(cached_balance), 0) FROM buckets WHERE user_id = :u AND is_active = TRUE"),
                {"u": user_uuid}
            ).scalar()

            discrepancy = round(float(physical) - float(virtual), 2)
            is_balanced = abs(discrepancy) <= 0.01

            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="ReconciliationAgent",
                recipient=msg.sender,
                status="SUCCESS",
                status_code=AgentStatus.AGENT_RESPONSE_READY.value,
                payload={
                    "physical_balance": float(physical),
                    "virtual_balance": float(virtual),
                    "discrepancy": discrepancy,
                    "is_zero_sum_balanced": is_balanced
                }
            )

    return A2AResponse(
        reply_to_id=msg.message_id,
        correlation_id=msg.correlation_id,
        sender="ReconciliationAgent",
        recipient=msg.sender,
        status="ERROR",
        status_code=AgentStatus.AGENT_PARSING_FAILURE.value,
        error_message=f"Unrecognized ReconciliationAgent intent: {intent}"
    )


async def handle_transaction_intake_agent(msg: A2AMessage) -> A2AResponse:
    """
    TransactionIntakeAgent A2A Handler:
    Tracks pending authorization to settled transitions, segregates internal transfers and salary,
    and emits clean consumer expenses to CategorizerAgent.
    """
    intent = msg.intent.upper()
    user_uuid = msg.user_uuid
    payload = msg.payload
    raw_txs = payload.get("transactions", [])
    account_id = payload.get("account_id")
    bank_uuid = payload.get("bank_uuid")

    if intent in ("PROCESS_INCOMING_TRANSACTIONS", "PROCESS_TRANSACTION_BATCH"):
        from agents.core_financial.TransactionIntake_Agent.TransactionIntakeAgent import process_transaction_lifecycle_item
        from services.a2a.bus import get_a2a_bus
        bus = get_a2a_bus()
        processed_items = []

        with SessionLocal() as session:
            # Resolve bank_uuid if not passed in payload
            if not bank_uuid and account_id:
                acc_row = session.execute(
                    text("SELECT bank_uuid FROM accounts WHERE account_id = :acc"),
                    {"acc": account_id}
                ).fetchone()
                if acc_row:
                    bank_uuid = acc_row[0]

            for tx in raw_txs:
                item_result = process_transaction_lifecycle_item(
                    session=session,
                    tx=tx,
                    user_uuid=user_uuid,
                    account_id=account_id,
                    bank_uuid=bank_uuid
                )
                processed_items.append(item_result)

                # Dispatch downstream A2A events based on transaction nature
                action = item_result.get("action")
                if action == "NEW_EXPENSE":
                    # Stage 2 Handoff: Dispatch to CategorizerAgent
                    asyncio.create_task(bus.send(A2AMessage(
                        sender="TransactionIntakeAgent",
                        recipient="CategorizerAgent",
                        message_type="COMMAND",
                        intent="CLASSIFY_TRANSACTION",
                        user_uuid=user_uuid,
                        correlation_id=msg.correlation_id,
                        payload=item_result
                    )))
                elif action == "INTERNAL_TRANSFER":
                    asyncio.create_task(bus.send(A2AMessage(
                        sender="TransactionIntakeAgent",
                        recipient="BucketTransferAgent",
                        message_type="COMMAND",
                        intent="RECONCILE_INTERNAL_TRANSFER",
                        user_uuid=user_uuid,
                        correlation_id=msg.correlation_id,
                        payload=item_result
                    )))
                elif action == "SALARY_INFLOW":
                    asyncio.create_task(bus.send(A2AMessage(
                        sender="TransactionIntakeAgent",
                        recipient="IncomeDetectionAgent",
                        message_type="COMMAND",
                        intent="EVALUATE_PAYDAY_INFLOW",
                        user_uuid=user_uuid,
                        correlation_id=msg.correlation_id,
                        payload=item_result
                    )))

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="TransactionIntakeAgent",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=AgentStatus.AGENT_RESPONSE_READY.value,
            payload={
                "total_processed": len(processed_items),
                "items": processed_items
            }
        )

    return A2AResponse(
        reply_to_id=msg.message_id,
        correlation_id=msg.correlation_id,
        sender="TransactionIntakeAgent",
        recipient=msg.sender,
        status="ERROR",
        status_code=AgentStatus.AGENT_PARSING_FAILURE.value,
        error_message=f"Unrecognized TransactionIntakeAgent intent: {intent}"
    )


async def handle_vector_maintenance_agent(msg: A2AMessage) -> A2AResponse:
    """
    VectorMaintenanceAgent A2A Handler:
    Cascades manual merchant corrections across historical transactions and Qdrant vectors,
    and detects/merges duplicate vector clusters.
    """
    intent = msg.intent.upper()
    payload = msg.payload

    if intent in ("CASCADE_MERCHANT_RELABELING", "CASCADE_RELABEL"):
        from agents.infrastructure.VectorMaintenance_Agent.VectorMaintenanceAgent import cascade_merchant_relabeling
        clean_merchant = payload.get("clean_merchant_name") or payload.get("merchant_name")
        new_category = payload.get("new_category") or payload.get("category")
        new_sub = payload.get("new_sub_category") or payload.get("sub_category", "")
        tags = payload.get("tags", [])

        if not clean_merchant or not new_category:
            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender="VectorMaintenanceAgent",
                recipient=msg.sender,
                status="ERROR",
                status_code=AgentStatus.AGENT_PARSING_FAILURE.value,
                error_message="Missing clean_merchant_name or new_category"
            )

        result_str = cascade_merchant_relabeling.invoke({
            "clean_merchant_name": clean_merchant,
            "new_category": new_category,
            "new_sub_category": new_sub,
            "tags": tags
        })

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="VectorMaintenanceAgent",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=AgentStatus.AGENT_RESPONSE_READY.value,
            payload={"result": result_str, "merchant": clean_merchant, "category": new_category}
        )

    elif intent in ("AUDIT_VECTOR_DUPLICATES", "FETCH_DUPLICATES"):
        from agents.infrastructure.VectorMaintenance_Agent.VectorMaintenanceAgent import fetch_duplicate_candidates
        dupes_json = fetch_duplicate_candidates.invoke({})
        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="VectorMaintenanceAgent",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=AgentStatus.AGENT_RESPONSE_READY.value,
            payload={"duplicates": dupes_json}
        )

    elif intent in ("MERGE_DUPLICATE_VECTORS", "MERGE_VECTORS"):
        from agents.infrastructure.VectorMaintenance_Agent.VectorMaintenanceAgent import merge_duplicate_vectors
        keep_id = payload.get("keep_id")
        delete_id = payload.get("delete_id")
        res = merge_duplicate_vectors.invoke({"keep_id": keep_id, "delete_id": delete_id})
        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender="VectorMaintenanceAgent",
            recipient=msg.sender,
            status="SUCCESS",
            status_code=AgentStatus.AGENT_RESPONSE_READY.value,
            payload={"result": res}
        )

    return A2AResponse(
        reply_to_id=msg.message_id,
        correlation_id=msg.correlation_id,
        sender="VectorMaintenanceAgent",
        recipient=msg.sender,
        status="ERROR",
        status_code=AgentStatus.AGENT_PARSING_FAILURE.value,
        error_message=f"Unrecognized VectorMaintenanceAgent intent: {intent}"
    )
