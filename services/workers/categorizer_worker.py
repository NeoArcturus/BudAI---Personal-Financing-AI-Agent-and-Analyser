import re
import os
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
async def classify_financial_data_wrapper(from_date: str, to_date: str, account_ids: list[str], state: Annotated[BudAIState, InjectedState]) -> str:
    """Categorize transactions for selected accounts."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("categorizer", "classify_financial_data", {"from_date": from_date, "to_date": to_date, "account_ids": account_ids, "user_uuid": user_uuid})

@tool
async def create_bargraph_chart_and_save_wrapper(account_ids: list[str], state: Annotated[BudAIState, InjectedState]) -> str:
    """Create bar chart of categories for selected accounts."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("categorizer", "create_bargraph_chart_and_save", {"account_ids": account_ids, "user_uuid": user_uuid})

@tool
async def create_pie_chart_and_save_wrapper(account_ids: list[str], state: Annotated[BudAIState, InjectedState]) -> str:
    """Create a pie chart representing the proportional distribution of spending categories."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("categorizer", "create_pie_chart_and_save", {"account_ids": account_ids, "user_uuid": user_uuid})

@tool
async def update_transaction_category_wrapper(transaction_uuid: str, corrected_category: str, state: Annotated[BudAIState, InjectedState]) -> str:
    """Manually update the category of a specific transaction."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("categorizer", "update_transaction_category", {"transaction_uuid": transaction_uuid, "corrected_category": corrected_category, "user_uuid": user_uuid})

@tool
async def retrain_categorization_model_wrapper(state: Annotated[BudAIState, InjectedState]) -> str:
    """Trigger the model to retrain based on corrected manual feedback."""
    user_uuid = state.get("user_uuid")
    return await bridge.call_tool("categorizer", "retrain_categorization_model", {"user_uuid": user_uuid})

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
    from services.mcp_tools.account_tools import get_connected_accounts
    user_uuid = state.get("user_uuid")
    return get_connected_accounts.invoke({"user_uuid": user_uuid})

@tool
async def ask_user(question: str) -> str:
    """Ask the user a question for clarification. Use this if you are unsure about which account to use for categorization."""
    return "Thinking..."

tools = [
    classify_financial_data_wrapper, 
    create_bargraph_chart_and_save_wrapper, 
    create_pie_chart_and_save_wrapper,
    update_transaction_category_wrapper,
    retrain_categorization_model_wrapper,
    export_advisory_state_wrapper,
    export_custom_statement_wrapper,
    get_connected_accounts_wrapper,
    ask_user
]

categorizer_agent_compiled = create_agent(
    model=llm,
    tools=tools,
    state_schema=BudAIState,
    system_prompt=f"""### ROLE: Specialist Financial Categorizer
You analyze and categorize transactions.
- Date: {current_date_str}

### CRITICAL STRICT ANTI-HALLUCINATION PROTOCOL ###
1. NO FABRICATION: You are strictly forbidden from fabricating data. Use ONLY data from tool DATA SUMMARY blocks.
2. TOOL EXECUTION IS MANDATORY: You must emit a valid JSON tool call to fetch data. You CANNOT roleplay or pretend to execute a tool in your output.
3. ADMIT IGNORANCE: If a tool returns no data, state "I do not have the data." Do not guess.
4. GBP ONLY: All financial values must use the £ symbol. No emojis.
5. REASONING VISIBILITY: You MUST ALWAYS provide an internal monologue wrapped explicitly inside <think> and </think> tags before taking ANY action, returning findings, or calling tools. Keep your <think> block EXTREMELY short (under 4 sentences). You MUST start your response exactly with `<think> Brief assessment: `

ROUTING (Use these tools):
- classify_financial_data_wrapper: Categorize transactions.
- create_bargraph_chart_and_save_wrapper: Bar chart of categories.
- create_pie_chart_and_save_wrapper: Pie chart of categories.
- update_transaction_category_wrapper: Manually update category.
- retrain_categorization_model_wrapper: Retrain model.
- export_advisory_state_wrapper: Save state.
- export_custom_statement_wrapper: Generate CSV.
- get_connected_accounts_wrapper: Get account IDs.
- ask_user: Ask clarifying questions.

FINAL AND MOST IMPORTANT INSTRUCTION:
You MUST start your VERY FIRST output character with the exact string: <think>
Do not say anything else before it.
""",
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"ask_user": True})
    ]
)

@tool("call_categorizer", description="Use this tool ONLY to break down user spending by category, classify merchants, or group user transactions.")
async def call_categorizer_agent(
    query: str, 
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[BudAIState, InjectedState]
):
    """Refined subagent tool for financial categorization."""
    user_uuid = state.get("user_uuid", "ea0e5c07-ab5b-4c14-9ad9-95a036b24637")
    result = await categorizer_agent_compiled.ainvoke({"messages": [{"role": "user", "content": query}], "user_uuid": user_uuid}, config=config)
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
