import os
import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
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

from datetime import datetime

current_date_str = datetime.now().strftime("%Y-%m-%d")
current_year_str = str(datetime.now().year)

base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
if not base_url.endswith("/v1"): 
    base_url = f"{base_url}/v1"
    
llm = ChatOpenAI(
    model="mlx-community/Qwen3.5-4B-4bit", 
    base_url=base_url, 
    api_key="budai-local", 
    temperature=0,
    streaming=False,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}}
)
bridge = MCPBridge()

from services.mcp_tools.external_tools import export_advisory_state, export_custom_statement

@tool
async def generate_hypothetical_scenario_wrapper(account_ids: list[str], days: int, injections: list[dict], state: Annotated[BudAIState, InjectedState]) -> str:
    """Generates a financial forecast based on a hypothetical scenario."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("forecaster", "generate_hypothetical_scenario", {"user_uuid": user_uuid, "account_ids": account_ids, "days": days, "injections": injections})

@tool
async def perform_currency_conversion_wrapper(amount: float, from_currency: str, to_currency: str, state: Annotated[BudAIState, InjectedState]) -> str:
    """Performs currency conversion using latest forex rates."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("macro", "perform_currency_conversion", {"amount": amount, "from_currency": from_currency, "to_currency": to_currency})

@tool
async def get_live_market_data_wrapper(assets: list[str], state: Annotated[BudAIState, InjectedState]) -> str:
    """Fetches real-time price and change for specific tickers."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("macro", "get_live_market_data", {"assets": assets})

@tool
async def get_financial_news_wrapper(query: str, state: Annotated[BudAIState, InjectedState]) -> str:
    """Fetches latest financial news/headlines for a topic."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("macro", "get_financial_news", {"query": query})

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
    """Ask the user a question for clarification regarding the scenario planning."""
    return "Thinking..."

tools = [
    generate_hypothetical_scenario_wrapper, 
    perform_currency_conversion_wrapper, 
    get_live_market_data_wrapper, 
    get_financial_news_wrapper, 
    export_advisory_state_wrapper,
    export_custom_statement_wrapper,
    get_connected_accounts_wrapper,
    ask_user
]

scenario_agent_compiled = create_agent(
    state_schema=BudAIState,
    model=llm,
    tools=tools,
    system_prompt=f"""### ROLE: Specialist Scenario Planner & Strategist
You model complex "what-if" life events.
- Date: {current_date_str}

### CRITICAL STRICT ANTI-HALLUCINATION PROTOCOL ###
1. NO FABRICATION: You are strictly forbidden from fabricating data. Use ONLY data from tool DATA SUMMARY blocks.
2. TOOL EXECUTION IS MANDATORY: You must emit a valid JSON tool call to fetch data. You CANNOT roleplay or pretend to execute a tool in your output.
3. ADMIT IGNORANCE: If a tool returns no data, state "I do not have the data." Do not guess.
4. GBP ONLY: All financial values must use the £ symbol. No emojis.
5. REASONING VISIBILITY: You MUST ALWAYS provide an internal monologue wrapped explicitly inside <think> and </think> tags before taking ANY action, returning findings, or calling tools. Keep your <think> block EXTREMELY short (under 4 sentences). You MUST start your response exactly with `<think> Brief assessment: `

ROUTING (Use these tools):
- generate_hypothetical_scenario_wrapper: Forecast scenario.
- perform_currency_conversion_wrapper: FX conversions.
- get_live_market_data_wrapper: Real-time price of assets.
- get_financial_news_wrapper: Latest financial news.
- export_advisory_state_wrapper: Save state.
- export_custom_statement_wrapper: Generate CSV.
- get_connected_accounts_wrapper: Get account IDs.
- ask_user: Ask clarifying questions.

THE USER'S FINANCIAL PROFILE WILL BE PROVIDED IN THE FIRST MESSAGE. USE IT FOR ALL INJECTIONS AND SCENARIOS.

FINAL AND MOST IMPORTANT INSTRUCTION:
You MUST start your VERY FIRST output character with the exact string: <think>
Do not say anything else before it.
""",
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"ask_user": True})
    ]
)

@tool("call_scenario_planner", description="Use this tool ONLY for complex, multi-event 'What-If' situations involving hypothetical financial injections or future life events.")
async def call_scenario_agent(
    query: str, 
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[BudAIState, InjectedState]
):
    """Refined subagent tool for Life Event Scenario Planning."""
    user_uuid = state.get("user_uuid", "ea0e5c07-ab5b-4c14-9ad9-95a036b24637")
    from services.profile_builder import ProfileBuilder
    profile_builder = ProfileBuilder(user_uuid)
    mrfp = await profile_builder.build_profile()
    
    query_with_context = f"<USER_PROFILE>\n{mrfp}\n</USER_PROFILE>\n\nUSER QUERY:\n{query}"
    
    result = await scenario_agent_compiled.ainvoke({"messages": [{"role": "user", "content": query_with_context}], "user_uuid": user_uuid}, config=config)
    raw_output = result["messages"][-1].content
    output = raw_output if isinstance(raw_output, str) else "".join([b if isinstance(b, str) else b.get("text", "") for b in raw_output if isinstance(b, (str, dict))])
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
