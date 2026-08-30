from services.mcp_tools.external_tools import export_advisory_state, export_custom_statement
from datetime import datetime
import re
import os
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from services.logger_setup import get_core_logger
from services.mcp_bridge import MCPBridge
from typing import Annotated
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from models.graph_state import BudAIState

logger = get_core_logger(__name__)

"""Returns a compiled subagent for financial forecasting."""
current_date_str = datetime.now().strftime("%Y-%m-%d")
current_year_str = str(datetime.now().year)

base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
if not base_url.endswith("/v1"):
    base_url = f"{base_url}/v1"

llm = ChatOpenAI(
    model="lmstudio-community/Qwen3.5-9B-GGUF", # Mac: model="mlx-community/Qwen3.5-4B-4bit",
    base_url=base_url,
    api_key="budai-local",
    temperature=0,
    streaming=False,
    max_tokens=4000,
    timeout=600,
    
)
bridge = MCPBridge()

@tool
async def generate_expense_forecast_wrapper(account_id: str, days: int, state: Annotated[BudAIState, InjectedState]) -> str:
    """Forecast future expenses for selected accounts."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("forecaster", "generate_expense_forecast", {"account_id": account_id, "days": days, "user_uuid": user_uuid})

@tool
async def generate_financial_forecast_wrapper(account_id: str, days: int, discipline_multiplier: float, state: Annotated[BudAIState, InjectedState]) -> str:
    """Forecast overall financial balance for selected accounts."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("forecaster", "generate_financial_forecast", {"account_id": account_id, "days": days, "discipline_multiplier": discipline_multiplier, "user_uuid": user_uuid})

@tool
async def generate_hypothetical_scenario_wrapper(account_id: str, days: int, injections: list[dict], state: Annotated[BudAIState, InjectedState]) -> str:
    """Generates a financial forecast based on a hypothetical scenario."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("forecaster", "generate_hypothetical_scenario", {"user_uuid": user_uuid, "account_id": account_id, "days": days, "injections": injections})

@tool
async def forecast_budget_impact_wrapper(account_id: str, days: int, state: Annotated[BudAIState, InjectedState]) -> str:
    """Run a detailed financial forecast incorporating the user's live budget pacing (Spend Velocity & Variance)."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("forecaster", "forecast_budget_impact", {"account_id": account_id, "days": days, "user_uuid": user_uuid})

@tool
async def export_advisory_state_wrapper(chart_type: str, raw_data: dict, ai_analysis: str, state: Annotated[BudAIState, InjectedState]) -> str:
    """Saves the current analytical state and AI insights to a persistent JSON file."""
    user_uuid = state.get("user_uuid")
    return export_advisory_state.invoke({"user_uuid": user_uuid, "chart_type": chart_type, "raw_data": raw_data, "ai_analysis": ai_analysis})

@tool
async def export_custom_statement_wrapper(ai_summary: str, state: Annotated[BudAIState, InjectedState]) -> str:
    """Generates a downloadable CSV transaction statement with embedded AI analysis."""
    user_uuid = state.get("user_uuid")
    return export_custom_statement.invoke({"user_uuid": user_uuid, "ai_summary": ai_summary})

@tool
async def get_connected_accounts_wrapper(state: Annotated[BudAIState, InjectedState]) -> str:
    """Use this tool to fetch the user's connected account IDs if you need to reference specific accounts."""
    user_uuid = state.get("user_uuid")
    from services.mcp_tools.account_tools import get_connected_accounts
    return get_connected_accounts.invoke({"user_uuid": user_uuid})

@tool
async def ask_user(question: str) -> str:
    """Ask the user a question for clarification. Use this if you are unsure about which account to use for forecasting."""
    return "Thinking..."

tools = [
    generate_expense_forecast_wrapper,
    generate_financial_forecast_wrapper,
    generate_hypothetical_scenario_wrapper,
    export_advisory_state_wrapper,
    export_custom_statement_wrapper,
    get_connected_accounts_wrapper,
    forecast_budget_impact_wrapper,
    ask_user
]

forecaster_agent_compiled = create_react_agent(
    model=llm,
    tools=tools,
    state_schema=BudAIState,
    prompt=f"""### ROLE: Specialist Financial Forecaster
You project future liquidity and scenarios.
- Date: {current_date_str}

### CRITICAL STRICT ANTI-HALLUCINATION PROTOCOL ###
1. NO FABRICATION: You are strictly forbidden from fabricating data. Use ONLY data from tool DATA SUMMARY blocks.
2. TOOL EXECUTION: You have access to specialized tools. You must use them to fetch data when required.
3. ADMIT IGNORANCE: If a tool returns no data, state "I do not have the data." Do not guess.
4. MULTI-CURRENCY: Respect the native currency returned by the tools (e.g., £, €, $). Do not force GBP. No emojis.
5. FRESH DATA: Always execute tools for fresh data. Never copy-paste numbers from chat history.

- YOUR SCOPE: Only projections where Year > {current_year_str}.

ROUTING (Use these tools):
- generate_expense_forecast_wrapper: Forecast future expenses.
- generate_financial_forecast_wrapper: Forecast overall financial balance.
- generate_hypothetical_scenario_wrapper: Forecast based on hypothetical events.
- export_advisory_state_wrapper: Save state.
- export_custom_statement_wrapper: Generate CSV.
- get_connected_accounts_wrapper: Get account IDs.
- ask_user: Ask clarifying questions.
- return_forecaster_findings: Final step to return data.
""",
    )

@tool("call_forecaster", description="Use this tool ONLY for mathematical projections of user balances or future user spending based on current patterns. Applies only to dates in the future.")
async def call_forecaster_agent(
    query: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[BudAIState, InjectedState]
):
    """Refined subagent tool for financial forecasting."""
    user_uuid = state.get("user_uuid", "ea0e5c07-ab5b-4c14-9ad9-95a036b24637")
    result = await forecaster_agent_compiled.ainvoke({"messages": [{"role": "user", "content": query}], "user_uuid": user_uuid})
    raw_output = result["messages"][-1].content
    output = raw_output if isinstance(raw_output, str) else "".join([b if isinstance(
        b, str) else b.get("text", "") for b in raw_output if isinstance(b, (str, dict))])
    cache_id, chart_type = None, None
    match = re.search(r'\[TRIGGER_([A-Z_]+):([^\]]+)\]', output)
    if match:
        chart_type = match.group(1)
        cache_id = match.group(2).split(':')[0]
        output = re.sub(r'\[TRIGGER_[A-Z_]+:[^\]]+\]', '', output).strip()

    return Command(update={
        "cache_id": cache_id,
        "chart_type": chart_type,
        "messages": [
            ToolMessage(content=output, tool_call_id=tool_call_id)
        ]
    })
