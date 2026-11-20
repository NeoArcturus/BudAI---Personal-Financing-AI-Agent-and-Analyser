import json
import uuid
from typing import Dict, Any, List, Optional, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from services.a2a.bus import get_a2a_bus
from services.a2a.protocol import A2AMessage
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

def resolve_user_uuid(state: Any) -> str:
    if isinstance(state, dict):
        return state.get("user_uuid") or ""
    return getattr(state, "user_uuid", "") or ""

def _ensure_workers_registered():
    from services.a2a.registry import get_agent_registry
    if not get_agent_registry().list_agents():
        from services.a2a.bootstrap import register_all_a2a_workers
        register_all_a2a_workers()

# ----------------- A2A Peer Delegation Tools -----------------

class HealthAgentInput(BaseModel):
    query_type: Optional[str] = Field(default="AUDIT_HEALTH_SCORE", description="'AUDIT_HEALTH_SCORE' or 'CHECK_EMERGENCY_RUNWAY'")

@tool("call_health_agent", args_schema=HealthAgentInput)
async def call_health_agent(
    query_type: str = "AUDIT_HEALTH_SCORE",
    state: Annotated[dict, InjectedState] = None
) -> str:
    """Delegate to HealthWorker via A2ABus for emergency runway, liquidity buffer audits, and financial wellness scores."""
    user_uuid = resolve_user_uuid(state)
    bus = get_a2a_bus()
    msg = A2AMessage(
        sender="AnalyserAgent",
        recipient="HealthWorker",
        message_type="QUERY",
        intent=query_type.upper() if query_type else "AUDIT_HEALTH_SCORE",
        payload={"query_type": query_type},
        user_uuid=user_uuid
    )
    resp = await bus.request_reply(msg)
    if resp.status == "SUCCESS":
        return json.dumps(resp.payload)
    return json.dumps({"error": resp.error_message or "Health audit failed", "status": resp.status})

class ForecasterAgentInput(BaseModel):
    days: int = Field(default=30, description="Forecast horizon in days (e.g. 30, 60, 90)")
    query_type: Optional[str] = Field(default="PROJECT_CASHFLOW", description="'PROJECT_CASHFLOW' or 'SIMULATE_STOCHASTIC_BALANCE'")

@tool("call_forecaster_agent", args_schema=ForecasterAgentInput)
async def call_forecaster_agent(
    days: int = 30,
    query_type: str = "PROJECT_CASHFLOW",
    state: Annotated[dict, InjectedState] = None
) -> str:
    """Delegate to ForecasterWorker via A2ABus for stochastic multi-path cashflow projections and Bates jump-diffusion corridors."""
    user_uuid = resolve_user_uuid(state)
    bus = get_a2a_bus()
    msg = A2AMessage(
        sender="AnalyserAgent",
        recipient="ForecasterWorker",
        message_type="QUERY",
        intent=query_type.upper() if query_type else "PROJECT_CASHFLOW",
        payload={"days": days},
        user_uuid=user_uuid
    )
    resp = await bus.request_reply(msg)
    if resp.status == "SUCCESS":
        return json.dumps(resp.payload)
    return json.dumps({"error": resp.error_message or "Forecast simulation failed", "status": resp.status})

class MarketAgentInput(BaseModel):
    category: Optional[str] = Field(default="CP00", description="COICOP category code or tag, e.g. 'CP00' (headline), 'CP01' (food), 'CP045' (energy)")

@tool("call_market_agent", args_schema=MarketAgentInput)
async def call_market_agent(
    category: str = "CP00",
    state: Annotated[dict, InjectedState] = None
) -> str:
    """Delegate to MarketWorker via A2ABus for macroeconomic indicators, ONS inflation benchmarks, and external market context."""
    user_uuid = resolve_user_uuid(state)
    bus = get_a2a_bus()
    msg = A2AMessage(
        sender="AnalyserAgent",
        recipient="MarketWorker",
        message_type="QUERY",
        intent="FETCH_MACRO_INFLATION",
        payload={"category": category},
        user_uuid=user_uuid
    )
    resp = await bus.request_reply(msg)
    if resp.status == "SUCCESS":
        return json.dumps(resp.payload)
    return json.dumps({"error": resp.error_message or "Market telemetry failed", "status": resp.status})

class CategorizerAgentInput(BaseModel):
    description: str = Field(description="Transaction or merchant description to classify")
    amount: Optional[float] = Field(default=0.0, description="Transaction amount")

@tool("call_categorizer_agent", args_schema=CategorizerAgentInput)
async def call_categorizer_agent(
    description: str,
    amount: float = 0.0,
    state: Annotated[dict, InjectedState] = None
) -> str:
    """Delegate to CategorizerAgent via A2ABus for merchant category taxonomy and ONS macro tags."""
    user_uuid = resolve_user_uuid(state)
    bus = get_a2a_bus()
    msg = A2AMessage(
        sender="AnalyserAgent",
        recipient="CategorizerAgent",
        message_type="QUERY",
        intent="CLASSIFY_TRANSACTION",
        payload={"description": description, "amount": amount},
        user_uuid=user_uuid
    )
    resp = await bus.request_reply(msg)
    if resp.status == "SUCCESS":
        return json.dumps(resp.payload)
    return json.dumps({"error": resp.error_message or "Classification failed", "status": resp.status})

class BucketAgentInput(BaseModel):
    intent: Optional[str] = Field(default="PROPOSE_REBALANCE", description="'PROPOSE_REBALANCE' or 'CHECK_REALITY_TRANSFERS'")

@tool("call_bucket_transfer_agent", args_schema=BucketAgentInput)
async def call_bucket_transfer_agent(
    intent: str = "PROPOSE_REBALANCE",
    state: Annotated[dict, InjectedState] = None
) -> str:
    """Delegate to BucketAgent via A2ABus for virtual capital allocations, bucket balances, and zero-sum rebalancing."""
    user_uuid = resolve_user_uuid(state)
    bus = get_a2a_bus()
    msg = A2AMessage(
        sender="AnalyserAgent",
        recipient="BucketAgent",
        message_type="QUERY",
        intent=intent.upper() if intent else "PROPOSE_REBALANCE",
        payload={},
        user_uuid=user_uuid
    )
    resp = await bus.request_reply(msg)
    if resp.status == "SUCCESS":
        return json.dumps(resp.payload)
    return json.dumps({"error": resp.error_message or "Bucket operation failed", "status": resp.status})

class AnomalyAgentInput(BaseModel):
    amount: float = Field(description="Transaction amount to evaluate")
    category: Optional[str] = Field(default="General", description="Category of transaction")

@tool("call_anomaly_detection_agent", args_schema=AnomalyAgentInput)
async def call_anomaly_detection_agent(
    amount: float,
    category: str = "General",
    state: Annotated[dict, InjectedState] = None
) -> str:
    """Delegate to AnomalyWorker via A2ABus for screening transactions for statistical surge variance and price shock outliers."""
    user_uuid = resolve_user_uuid(state)
    bus = get_a2a_bus()
    msg = A2AMessage(
        sender="AnalyserAgent",
        recipient="AnomalyWorker",
        message_type="QUERY",
        intent="EVALUATE_TRANSACTION_ANOMALY",
        payload={"amount": amount, "category": category},
        user_uuid=user_uuid
    )
    resp = await bus.request_reply(msg)
    if resp.status == "SUCCESS":
        return json.dumps(resp.payload)
    return json.dumps({"error": resp.error_message or "Anomaly evaluation failed", "status": resp.status})

@tool("call_subscription_detection_agent")
async def call_subscription_detection_agent(
    state: Annotated[dict, InjectedState] = None
) -> str:
    """Delegate to SubscriptionWorker via A2ABus to audit active recurring billing, temporal rhythms, and hidden price hikes."""
    user_uuid = resolve_user_uuid(state)
    bus = get_a2a_bus()
    msg = A2AMessage(
        sender="AnalyserAgent",
        recipient="SubscriptionWorker",
        message_type="QUERY",
        intent="SCAN_RECURRING_PAYMENTS",
        payload={},
        user_uuid=user_uuid
    )
    resp = await bus.request_reply(msg)
    if resp.status == "SUCCESS":
        return json.dumps(resp.payload)
    return json.dumps({"error": resp.error_message or "Subscription scan failed", "status": resp.status})

class ScenarioAgentInput(BaseModel):
    shock_amount: float = Field(default=500.0, description="Magnitude of the unexpected expense shock")
    category: Optional[str] = Field(default="General", description="Expense shock domain")

@tool("call_scenario_agent", args_schema=ScenarioAgentInput)
async def call_scenario_agent(
    shock_amount: float = 500.0,
    category: str = "General",
    state: Annotated[dict, InjectedState] = None
) -> str:
    """Delegate to ScenarioWorker via A2ABus for computing structural financial scenarios and emergency stress tests."""
    user_uuid = resolve_user_uuid(state)
    bus = get_a2a_bus()
    msg = A2AMessage(
        sender="AnalyserAgent",
        recipient="ScenarioWorker",
        message_type="QUERY",
        intent="SIMULATE_EXPENSE_SHOCK",
        payload={"shock_amount": shock_amount, "category": category},
        user_uuid=user_uuid
    )
    resp = await bus.request_reply(msg)
    if resp.status == "SUCCESS":
        return json.dumps(resp.payload)
    return json.dumps({"error": resp.error_message or "Scenario simulation failed", "status": resp.status})

# ----------------- Support Tools & Wrappers -----------------

@tool("call_memory_service")
async def call_memory_service(
    query: str,
    state: Annotated[dict, InjectedState] = None
) -> str:
    """Retrieve qualitative user financial goals and preferences from the persistent memory service."""
    user_uuid = resolve_user_uuid(state)
    try:
        from services.memory_service import memory_service
        results = await memory_service.get_memories(user_uuid, query=query, limit=5)
        return json.dumps(results) if results else "No qualitative memory found."
    except Exception as e:
        logger.warning(f"Memory service call failed: {e}")
        return "No qualitative memory found."

@tool("get_connected_accounts_orchestrator")
def get_connected_accounts_orchestrator(state: Annotated[dict, InjectedState] = None) -> str:
    """Fetch the user's connected account details (like account IDs, bank names, balances) from the verified database."""
    from services.mcp_tools.account_tools import get_connected_accounts
    user_uuid = resolve_user_uuid(state)
    return get_connected_accounts.invoke({"user_uuid": user_uuid})

@tool("get_liability_horizon_orchestrator")
def get_liability_horizon_orchestrator(state: Annotated[dict, InjectedState] = None) -> str:
    """Fetch the user's Liability Horizon (Upcoming Direct Debits, Standing Orders, and Pending Transactions)."""
    from services.mcp_tools.core_query_tools import get_liability_horizon
    user_uuid = resolve_user_uuid(state)
    return get_liability_horizon.invoke({"user_uuid": user_uuid})

@tool("get_user_lifestyle_profile_wrapper")
def get_user_lifestyle_profile_wrapper(state: Annotated[dict, InjectedState] = None) -> str:
    """Retrieves the user's Macro-Persona and their top HDBSCAN micro-lifestyles."""
    from services.mcp_tools.lifestyle_tools import get_user_lifestyle_profile
    user_uuid = resolve_user_uuid(state)
    return get_user_lifestyle_profile.invoke({"user_uuid": user_uuid})

@tool("get_semantic_anomalies_wrapper")
def get_semantic_anomalies_wrapper(state: Annotated[dict, InjectedState] = None) -> str:
    """Fetches recent transactions flagged as semantic anomalies."""
    from services.mcp_tools.lifestyle_tools import get_semantic_anomalies
    user_uuid = resolve_user_uuid(state)
    return get_semantic_anomalies.invoke({"user_uuid": user_uuid})

@tool("get_lifestyle_trajectory_wrapper")
def get_lifestyle_trajectory_wrapper(state: Annotated[dict, InjectedState] = None) -> str:
    """Calculates how the user's cluster density has shifted over the last 3 months."""
    from services.mcp_tools.lifestyle_tools import get_lifestyle_trajectory
    user_uuid = resolve_user_uuid(state)
    return get_lifestyle_trajectory.invoke({"user_uuid": user_uuid})

@tool("benchmark_persona_budget_wrapper")
def benchmark_persona_budget_wrapper(state: Annotated[dict, InjectedState] = None) -> str:
    """Compares the user's spending against statistical averages for their assigned Macro-Persona."""
    from services.mcp_tools.lifestyle_tools import benchmark_persona_budget
    user_uuid = resolve_user_uuid(state)
    return benchmark_persona_budget.invoke({"user_uuid": user_uuid})

@tool("predict_impulse_vulnerability_wrapper")
def predict_impulse_vulnerability_wrapper(state: Annotated[dict, InjectedState] = None) -> str:
    """Analyzes time/day vectors to identify statistical impulse buying windows."""
    from services.mcp_tools.lifestyle_tools import predict_impulse_vulnerability
    user_uuid = resolve_user_uuid(state)
    return predict_impulse_vulnerability.invoke({"user_uuid": user_uuid})

@tool("get_upcoming_subscriptions_wrapper")
def get_upcoming_subscriptions_wrapper(state: Annotated[dict, InjectedState] = None) -> str:
    """Retrieves detected recurring subscriptions, bills, and any hidden price hikes."""
    from services.mcp_tools.lifestyle_tools import get_upcoming_subscriptions
    user_uuid = resolve_user_uuid(state)
    return get_upcoming_subscriptions.invoke({"user_uuid": user_uuid})

class GenerateUIChartInput(BaseModel):
    type: Literal["bar", "line", "pie", "radar", "doughnut"] = Field(description="Chart type")
    title: str = Field(description="Chart title")
    subtitle: Optional[str] = Field(default=None, description="Chart subtitle")
    labels: List[str] = Field(description="Labels for the chart axes or slices")
    datasets: List[Dict[str, Any]] = Field(description="Data series for the chart")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Optional chart configuration")

@tool("generate_ui_chart", args_schema=GenerateUIChartInput)
def generate_ui_chart(
    type: str,
    title: str,
    labels: List[str],
    datasets: List[Dict[str, Any]],
    subtitle: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None
) -> str:
    """Render a dynamic chart in the user interface to visually illustrate financial trends, spending allocations, or forecasts."""
    return f"Chart '{title}' rendered successfully."

@tool("ask_user")
async def ask_user(question: str) -> str:
    """Ask the user a question for clarification or more information when an inquiry is ambiguous."""
    return "Thinking..."
