import asyncio
from datetime import datetime
import re
import json
import uuid
from models.database_models import ChatHistory, ChatSession, Account, Bank
from config import SessionLocal
from services.workers.analyser_worker import call_analyser_agent
from services.workers.forecaster_worker import call_forecaster_agent
from services.workers.categorizer_worker import call_categorizer_agent
from services.workers.health_worker import call_health_agent
from services.workers.memory_worker import call_memory_agent
from services.workers.market_worker import call_market_agent
from services.workers.scenario_worker import call_scenario_agent

from models.graph_state import BudAIState
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Any
import os
from services.mcp_bridge import MCPBridge
from services.mcp_tools.external_tools import export_advisory_state
from services.logger_setup import get_core_logger
from langchain_core.runnables import RunnableConfig

logger = get_core_logger("orchestrator_graph")

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")


@tool
async def ask_user(question: str) -> str:
    """Ask the user a question for clarification or more information. Use this if the user's request is ambiguous or if you need more details to decide which specialist tool to call."""
    return "Thinking..."

from langgraph.prebuilt import InjectedState
from typing import Annotated
@tool
def get_connected_accounts_orchestrator(state: Annotated[BudAIState, InjectedState]) -> str:
    """Use this tool to fetch the user's connected account details (like account IDs) from the database."""
    from services.mcp_tools.account_tools import get_connected_accounts
    user_uuid = state.get("user_uuid", "ea0e5c07-ab5b-4c14-9ad9-95a036b24637")
    return get_connected_accounts.invoke({"user_uuid": user_uuid})


def get_session_history(user_uuid: str, session_id: Optional[str] = None):
    """Retrieves session history from database."""
    history = []
    try:
        with SessionLocal() as session:
            query = session.query(ChatHistory).filter(
                ChatHistory.user_uuid == user_uuid)
            if session_id:
                query = query.filter(ChatHistory.session_id == session_id)
            else:
                query = query.filter(ChatHistory.session_id == None)
            records = query.order_by(ChatHistory.timestamp.asc()).all()
            for r in records:
                if r.role == "user":
                    history.append(HumanMessage(content=r.content))
                else:
                    history.append(AIMessage(content=r.content))
    except Exception as e:
        logger.error(f"Failed to fetch session history: {e}")
    return history[-6:]


def ensure_string(content: Any) -> str:
    """Robustly converts LLM content to a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join([b.get("text", b.get("content", "")) if isinstance(b, dict) else str(b) for b in content])
    return str(content)


def strip_thinking(content: Any) -> str:
    """Removes thinking tags from output."""
    text = ensure_string(content)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


async def generate_session_title(session_id: str, first_msg: str):
    """Generates an automatic session title."""
    try:
        llm = ChatOpenAI(
            model="mlx-community/Qwen3.5-4B-4bit",
            base_url=OLLAMA_BASE_URL,
            api_key="budai-local",
            temperature=0
        )
        res = await llm.ainvoke(f"Professional title for: '{first_msg}'. 3-4 words.")
        title = res.content.strip().replace('"', '')
        with SessionLocal() as session:
            db_session = session.query(ChatSession).filter_by(
                session_id=session_id).first()
            if db_session:
                db_session.title = title
                session.commit()
    except Exception:
        pass


async def execute_chat_graph_async(initial_state: dict):
    """Initializes the database records for a chat session."""
    user_uuid = initial_state['user_uuid']
    session_id = initial_state.get('session_id')

    try:
        with SessionLocal() as session:
            db_session = None
            if session_id:
                db_session = session.query(ChatSession).filter_by(
                    session_id=session_id, user_uuid=user_uuid).first()

            if not db_session:
                new_session_id = session_id if session_id else str(
                    uuid.uuid4())
                db_session = ChatSession(
                    session_id=new_session_id, user_uuid=user_uuid, title="Analyzing...")
                session.add(db_session)
                session.commit()
                initial_state['session_id'] = new_session_id
                session_id = new_session_id
                asyncio.create_task(generate_session_title(
                    session_id, initial_state['user_input']))
            else:
                db_session.last_updated = datetime.utcnow()
                session.commit()

            new_msg = ChatHistory(user_uuid=user_uuid, session_id=session_id,
                                  role="user", content=initial_state['user_input'])
            session.add(new_msg)
            session.commit()
    except Exception as e:
        logger.error(f"DB failed: {e}")


supervisor_llm = ChatOpenAI(
    model="mlx-community/Qwen3.5-4B-4bit",
    base_url=OLLAMA_BASE_URL,
    api_key="budai-local",
    temperature=0.1,
    streaming=True,
    max_tokens=4096,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}}
)

supervisor_tools = [
    call_analyser_agent,
    call_forecaster_agent,
    call_categorizer_agent,
    call_health_agent,
    call_memory_agent,
    call_market_agent,
    call_scenario_agent,
    ask_user,
    get_connected_accounts_orchestrator
]

current_date_str = datetime.now().strftime("%Y-%m-%d")
current_year_str = str(datetime.now().year)

budai_app = create_agent(
    model=supervisor_llm,
    tools=supervisor_tools,
    state_schema=BudAIState,
    system_prompt=f"""### ROLE: Financial Advisor (BudAI)
You are BudAI, a personal finance advisor.
- Date: {current_date_str}

### CRITICAL STRICT ANTI-HALLUCINATION PROTOCOL ###
1. NO FABRICATION: You are strictly forbidden from fabricating data. Use ONLY data from tool DATA SUMMARY blocks.
2. TOOL EXECUTION IS MANDATORY: You must emit a valid JSON tool call to fetch data. You CANNOT roleplay or pretend to execute a tool in your output.
3. ADMIT IGNORANCE: If a tool returns no data, state "I do not have the data." Do not guess.
4. GBP ONLY: All financial values must use the £ symbol. No emojis.
5. REASONING VISIBILITY: You MUST ALWAYS provide an internal monologue wrapped explicitly inside <think> and </think> tags before taking ANY action or responding. Keep your <think> block EXTREMELY short (under 4 sentences). You MUST start your response exactly with `<think> Brief assessment: `
6. FRESH DATA: Always execute tools for fresh data. Never copy-paste numbers from chat history.
7. CHART TRIGGERS: If a tool outputs a tag like `[TRIGGER_...:CACHE_...]`, copy it EXACTLY as the very last line of your text response. Do not modify it.
8. SINGLE ACCOUNT: Default to the connected account if only 1 exists. Ask user if multiple exist and query is ambiguous.

ROUTING (Use these tools):
- call_analyser_agent: historical transactions, spending totals, cash flow.
- call_forecaster_agent: future projections.
- call_categorizer_agent: categorization, merchant grouping.
- call_health_agent: financial health, emergency funds.
- call_memory_agent: past preferences, qualitative facts.
- call_market_agent: real-time/historical market data (stocks, FX).
- call_scenario_agent: complex 'What-If' scenarios.

FINAL AND MOST IMPORTANT INSTRUCTION:
You MUST start your VERY FIRST output character with the exact string: <think>
Do not say anything else before it.
""",
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"ask_user": True})
    ],
    checkpointer=InMemorySaver()
)
