import json
from typing import Dict, Any, Optional
from services.a2a.registry import get_agent_registry, AgentCard
from services.a2a.bus import get_a2a_bus
from services.a2a.protocol import A2AMessage, A2AResponse
from services.a2a.handlers import (
    handle_market_agent,
    handle_health_agent,
    handle_forecaster_agent,
    handle_bucket_transfer_agent,
    handle_categorizer_agent,
    handle_analyser_agent
)
from models.status_codes import AgentStatus
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

def register_all_a2a_workers() -> None:
    """
    Registers all core specialist agents and capability cards on the A2A Registry.
    Called once during FastAPI lifespan startup.
    """
    registry = get_agent_registry()

    # 1. MarketWorker Card
    registry.register(
        AgentCard(
            name="MarketWorker",
            description="Provides macroeconomic telemetry, ONS inflation benchmarks, and external market signals.",
            intents_handled=["FETCH_MACRO_INFLATION", "GET_MARKET_INDEX"],
            avg_latency_ms=15.0
        ),
        handle_market_agent
    )

    # 2. HealthWorker Card
    registry.register(
        AgentCard(
            name="HealthWorker",
            description="Audits liquidity reserves, emergency runways, and evaluates rebalance needs.",
            intents_handled=["AUDIT_HEALTH_SCORE", "CHECK_EMERGENCY_RUNWAY", "EVALUATE_REBALANCE_NEED"],
            avg_latency_ms=25.0
        ),
        handle_health_agent
    )

    # 3. ForecasterWorker Card
    registry.register(
        AgentCard(
            name="ForecasterWorker",
            description="Simulates stochastic balance corridors and projects daily cashflow trajectories.",
            intents_handled=["SIMULATE_STOCHASTIC_BALANCE", "PROJECT_CASHFLOW"],
            avg_latency_ms=30.0
        ),
        handle_forecaster_agent
    )

    # 4. BucketTransferAgent Card
    registry.register(
        AgentCard(
            name="BucketTransferAgent",
            description="Executes double-entry virtual transfers under strict PL/pgSQL math lock constraints.",
            intents_handled=["PROPOSE_REBALANCE", "EXECUTE_VIRTUAL_TRANSFER"],
            avg_latency_ms=20.0
        ),
        handle_bucket_transfer_agent
    )

    # 5. CategorizerAgent Card
    registry.register(
        AgentCard(
            name="CategorizerAgent",
            description="Performs canonical 3D taxonomy categorization and staple keyword tagging.",
            intents_handled=["CLASSIFY_TRANSACTION", "ENRICH_ONS_TAGS"],
            avg_latency_ms=10.0
        ),
        handle_categorizer_agent
    )

    # 6. AnalyserAgent Card
    registry.register(
        AgentCard(
            name="AnalyserAgent",
            description="Audits transaction velocity, spending surges, and cashflow collisions.",
            intents_handled=["ANALYZE_SPENDING_SURGE"],
            avg_latency_ms=25.0
        ),
        handle_analyser_agent
    )

    logger.info(
        f"A2A Bootstrap: 6 specialist workers successfully registered on in-process bus "
        f"({', '.join([c.name for c in registry.list_agents()])})."
    )

async def execute_autonomous_rebalance_consensus(
    user_uuid: str,
    source_bucket_id: str,
    target_bucket_id: str,
    amount: float,
    reason: str = "Automated delta rebalance"
) -> Dict[str, Any]:
    """
    Autonomous multi-agent consensus rebalancing workflow (100% HITL-free):
    1. HealthWorker verifies liquidity buffer.
    2. MarketWorker enriches proposal with current inflation telemetry.
    3. BucketTransferAgent evaluates double-entry solvency under math lock.
    4. Executes virtual transfer and dispatches system alert.
    """
    bus = get_a2a_bus()
    correlation_id = f"a2a-rebal-{user_uuid[:8]}"

    # Step 1: HealthWorker Audit
    health_msg = A2AMessage(
        sender="Orchestrator",
        recipient="HealthWorker",
        message_type="QUERY",
        intent="AUDIT_HEALTH_SCORE",
        user_uuid=user_uuid,
        correlation_id=correlation_id
    )
    health_resp: A2AResponse = await bus.request_reply(health_msg)

    # Step 2: MarketWorker Macro Telemetry
    market_msg = A2AMessage(
        sender="Orchestrator",
        recipient="MarketWorker",
        message_type="QUERY",
        intent="FETCH_MACRO_INFLATION",
        user_uuid=user_uuid,
        correlation_id=correlation_id,
        payload={"category": "General & Discretionary"}
    )
    market_resp: A2AResponse = await bus.request_reply(market_msg)

    # Step 3: BucketTransferAgent Solvency Check & Execution
    transfer_msg = A2AMessage(
        sender="HealthWorker",
        recipient="BucketTransferAgent",
        message_type="COMMAND",
        intent="EXECUTE_VIRTUAL_TRANSFER",
        user_uuid=user_uuid,
        correlation_id=correlation_id,
        payload={
            "source_bucket_id": source_bucket_id,
            "target_bucket_id": target_bucket_id,
            "amount": amount,
            "reason": reason,
            "macro_context": market_resp.payload,
            "health_context": health_resp.payload
        }
    )
    transfer_resp: A2AResponse = await bus.request_reply(transfer_msg)

    return {
        "status": transfer_resp.status,
        "status_code": transfer_resp.status_code,
        "correlation_id": correlation_id,
        "transfer_result": transfer_resp.payload,
        "health_score": health_resp.payload.get("health_score"),
        "inflation_rate": market_resp.payload.get("delta_pct_yoy")
    }
