import asyncio
from datetime import datetime
import re
import json
import uuid
from models.database_models import ChatHistory, ChatSession, Account, Bank
from config import SessionLocal
from services.workers import analyser_worker, forecaster_worker, categorizer_worker, health_worker, memory_worker, market_worker
from models.graph_state import BudAIState
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler, StdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
import os
from services.mcp_bridge import MCPBridge
from services.mcp_tools.external_tools import export_advisory_state
from services.logger_setup import get_core_logger

logger = get_core_logger("orchestrator_graph")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")

class RouteDecision(BaseModel):
    reasoning: str = Field(..., description="Step-by-step reasoning explaining why this worker was selected.")
    selected_worker: Literal["analyser", "forecaster", "categorizer", "health", "memory", "market", "general", "account_clarifier"] = Field(...)

router_llm = ChatOpenAI(
    model="mlx-community/Qwen2.5-7B-Instruct-4bit",
    base_url=OLLAMA_BASE_URL,
    api_key="budai-local",
    temperature=0
)

current_date = datetime.now().strftime("%Y-%m-%d")
current_year = datetime.now().year

def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def get_session_history(user_uuid: str, session_id: Optional[str] = None):
    history = []
    try:
        with SessionLocal() as session:
            query = session.query(ChatHistory).filter(ChatHistory.user_uuid == user_uuid)
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
    return history

async def ingest_context_node(state: BudAIState):
    bridge = MCPBridge()
    try:
        local_rules = await asyncio.to_thread(bridge.read_user_rules)
    except Exception as e:
        logger.error(f"Failed to ingest local rules: {e}")
        local_rules = {}
    return {"local_rules": local_rules}

async def intent_router_node(state: BudAIState):
    chat_history = get_session_history(state['user_uuid'], state.get('session_id'))
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an INVISIBLE, ROBOTIC query routing mechanism. YOUR ONLY purpose is to classify the user's latest query.
         ROUTING LOGIC:
         - Breakdown/categories -> categorizer
         - Past/historical data (Year <= {current_year}) -> analyser
         - Future predictions (Year > {current_year}) -> forecaster
         - Survival/Debt/Health -> health
         - Memory/Preferences -> memory
         - Economy/Markets/News/FX -> market
         - Non-data conversational questions (or hypothetical/survival/travel advice) -> general

         AMBIGUITY CHECK:
         If the query requires specific bank data (categorizer, analyser, forecaster) AND the user has NOT specified which bank(s) to use, you MUST route to 'account_clarifier'.
         Example 1: "How much did I spend?" -> account_clarifier
         Example 2: "Show my Wise expenses" -> analyser
         Example 3: "How is my overall health?" -> health (health is global, no clarifier needed)
         """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Classify this query: {input}")
    ])
    structured_router_llm = router_llm.with_structured_output(RouteDecision)
    chain = prompt | structured_router_llm
    try:
        is_pinned = state.get('active_account_id') and state['active_account_id'] != "ALL"
        
        result = await chain.ainvoke({"input": state['user_input'], "current_year": current_year, "chat_history": chat_history})
        decision = result.selected_worker
        
        data_workers = ["analyser", "forecaster", "categorizer"]
        if decision in data_workers and not is_pinned:
            with SessionLocal() as session:
                rows = session.query(Account.account_id, Bank.bank_name).join(Bank).filter(Account.user_uuid == state['user_uuid']).all()
            
            if len(rows) == 1:
                state['user_input'] += f" (Auto-selected account: {rows[0][1]}, ID: {rows[0][0]})"
            else:
                bank_names = [r[1].lower() for r in rows if r[1]]
                vague = True
                user_input_lower = state['user_input'].lower()
                matched_names = []
                for name in bank_names:
                    if name in user_input_lower:
                        matched_names.append(name)
                        vague = False
                
                if decision == "forecaster" and len(matched_names) != 1:
                    vague = True
                
                if vague:
                    logger.info(f"Ambiguity detected for {decision}. Routing to account_clarifier.")
                    decision = "account_clarifier"
            
    except Exception as e:
        logger.error(f"LLM Routing Call Failed: {e}")
        decision = "general"
    return {"selected_worker": decision}

def route_to_worker(state: BudAIState):
    return state["selected_worker"]

def create_account_clarifier_node(q: asyncio.Queue):
    async def account_clarifier_node(state: BudAIState):
        logger.info("Triggering account selector modal")
        accounts_list = []
        try:
            with SessionLocal() as session:
                rows = session.query(Account.account_id, Bank.bank_name).join(Bank).filter(Account.user_uuid == state['user_uuid']).all()
                accounts_list = [{"id": r[0], "name": r[1]} for r in rows]
        except Exception as e:
            logger.error(f"Failed to fetch accounts: {e}")

        tool_call_payload = {
            "type": "tool_call",
            "toolName": "render_account_selector",
            "args": {
                "available_accounts": accounts_list
            }
        }
        
        msg = "I see you have multiple accounts. Which one(s) should I look into for you?"
        q.put_nowait(msg)
        q.put_nowait(f"__DATA__{json.dumps({'type': 'trigger_account_selector'})}")
        q.put_nowait(f"__TOOL_CALL__{json.dumps(tool_call_payload)}")
        
        return {"final_response": msg}
    return account_clarifier_node

async def ui_router_node(state: BudAIState):
    return {"ui_trigger_tag": ""}

def create_response_generator_node(q: asyncio.Queue):
    async def response_generator_node(state: BudAIState):
        class StreamHandler(BaseCallbackHandler):
            def __init__(self):
                self.is_thinking = False
            def on_llm_new_token(self, token: str, **kwargs) -> None:
                if "<think>" in token:
                    self.is_thinking = True
                    return
                if "</think>" in token:
                    self.is_thinking = False
                    return
                if not self.is_thinking:
                    q.put_nowait(token)
            def finalize(self):
                pass

        stream_handler = StreamHandler()
        persona_llm = ChatOpenAI(
            model="mlx-community/Qwen2.5-7B-Instruct-4bit",
            base_url=OLLAMA_BASE_URL,
            api_key="budai-local",
            temperature=0.1,
            streaming=True,
            callbacks=[stream_handler]
        )
        chat_history = get_session_history(state['user_uuid'], state.get('session_id'))
        from services.profile_builder import ProfileBuilder
        profile_builder = ProfileBuilder(state['user_uuid'])
        mrfp = await profile_builder.build_profile()

        prompt_rules = """
            CRITICAL OPERATIONAL MANDATES:
            1. ZERO HALLUCINATION: Never invent financial figures. However, you MUST use the numerical projections provided in the DATA SUMMARY as verified facts.
            2. DATA PRIMACY: If the DATA SUMMARY provides an 'Expected £X' or 'Projected £Y' value, that IS your data.
            3. NO EMOJIS: Use plain text only.
            4. GBP ONLY: All currency must be in GBP (£).
            5. TECHNICAL PRECISION: Use exact figures provided.
"""
        if state.get('selected_worker') == 'general':
            prompt_rules = """
            CRITICAL OPERATIONAL MANDATES:
            1. ZERO HALLUCINATION: Never invent financial figures.
            2. HYPOTHETICAL/GENERAL: Answer based on the user's profile and general logic. DO NOT apologize for missing data.
            3. NO EMOJIS: Strictly forbidden.
            4. GBP ONLY: Always use £.
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are BudAI, a precise senior financial advisor.\n\n{prompt_rules}\n\n### USER PROFILE\n{mrfp}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "User: {input}\nData Summary: {worker_summary}")
        ])
        
        chain = prompt | persona_llm
        try:
            result = await chain.ainvoke({"input": state['user_input'], "worker_summary": state.get('worker_summary', 'No data.'), "chat_history": chat_history})
        except Exception as e:
            logger.error(f"Persona LLM Failed: {e}")
            raise e

        stream_handler.finalize()
        warm_text = strip_thinking(result.content)
        cache_id = state.get('cache_id')
        chart_type = state.get('chart_type')

        try:
            with SessionLocal() as session:
                if warm_text:
                    new_msg = ChatHistory(user_uuid=state['user_uuid'], session_id=state.get('session_id'), role="assistant", content=warm_text)
                    session.add(new_msg)
                    session.commit()
                    if state.get('raw_data') and chart_type:
                        await asyncio.to_thread(export_advisory_state.invoke, {"user_uuid": state['user_uuid'], "chart_type": chart_type, "raw_data": state['raw_data'], "ai_analysis": warm_text})
        except Exception as db_e:
            logger.error(f"Failed to store output: {db_e}")

        if cache_id and chart_type:
            tool_call_payload = {"type": "tool_call", "toolName": "render_ui_chart", "args": {"chart_type": chart_type, "cache_id": cache_id}}
            q.put_nowait(f"__TOOL_CALL__{json.dumps(tool_call_payload)}")
            q.put_nowait(f"__DATA__{json.dumps({'type': 'global_refresh_signal', 'chart_type': chart_type})}")
        return {"final_response": warm_text}
    return response_generator_node

def create_explainer_node(q: asyncio.Queue):
    async def explainer_node(state: BudAIState):
        class StreamHandler(BaseCallbackHandler):
            def __init__(self):
                self.is_thinking = False
            def on_llm_new_token(self, token: str, **kwargs) -> None:
                if "<think>" in token:
                    self.is_thinking = True
                    return
                if "</think>" in token:
                    self.is_thinking = False
                    return
                if not self.is_thinking:
                    q.put_nowait(token)
            def finalize(self):
                pass

        stream_handler = StreamHandler()
        persona_llm = ChatOpenAI(
            model="mlx-community/Qwen2.5-7B-Instruct-4bit",
            base_url=OLLAMA_BASE_URL,
            api_key="budai-local",
            temperature=0.1,
            streaming=True,
            callbacks=[stream_handler]
        )
        chat_history = get_session_history(state['user_uuid'], state.get('session_id'))
        current_date_str = datetime.now().strftime("%Y-%m-%d")
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are BudAI, a senior financial data analyst. Your ONLY goal is to explain the provided JSON data accurately.

            CRITICAL GROUNDING RULES:
            1. STRICT DATA BOUNDARY: Use ONLY numerical values from JSON. If missing, state "I do not have data on that."
            2. ZERO EXTRAPOLATION: Do not guess.
            3. TIME AWARENESS: Compare provided dates to Today ({current_date}).
            4. NO HALLUCINATION: 100% accuracy required.
            5. NO EMBELLISHMENT: Professional observations only.
            6. GBP ONLY: Use £.
            7. NO EMOJIS: Strictly forbidden.

            RESPONSE STRUCTURE:
            - 2-3 sentence technical summary.
            - 2-3 observations.
            - exactly three follow-up actions in JSON markdown.

            VALID ACTIONS: 'classify_financial_data', 'plot_health_radar', 'historical_monthly', 'expense_forecast', 'balance_forecast'.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Today's Date: {current_date}\nTool Used: {chart_type}\nRaw Data: {raw_data}")
        ])
        
        chain = prompt | persona_llm
        try:
            result = await chain.ainvoke({
                "chart_type": state.get('chart_type', 'UNKNOWN'),
                "raw_data": json.dumps(state.get('raw_data', {})),
                "current_date": current_date_str,
                "chat_history": chat_history
            })
        except Exception as e:
            logger.error(f"Explainer LLM Failed: {e}")
            raise e
        stream_handler.finalize()
        warm_text = strip_thinking(result.content)
        try:
            with SessionLocal() as session:
                new_msg = ChatHistory(user_uuid=state['user_uuid'], session_id=state.get('session_id'), role="assistant", content=warm_text)
                session.add(new_msg)
                session.commit()
        except Exception as db_e:
            logger.error(f"Failed to store output: {db_e}")
        return {"final_response": warm_text}
    return explainer_node

def route_initial(state: BudAIState):
    return "explainer" if state.get("is_explanation") else "intent_router"

async def generate_session_title(session_id: str, first_msg: str, q: Optional[asyncio.Queue] = None):
    try:
        llm = ChatOpenAI(
            model="mlx-community/Qwen2.5-7B-Instruct-4bit",
            base_url=OLLAMA_BASE_URL,
            api_key="budai-local",
            temperature=0
        )
        res = await llm.ainvoke(f"Professional title for: '{first_msg}'. 3-4 words.")
        title = res.content.strip().replace('"', '')
        with SessionLocal() as session:
            db_session = session.query(ChatSession).filter_by(session_id=session_id).first()
            if db_session:
                db_session.title = title
                session.commit()
                if q:
                    q.put_nowait(f"__DATA__{json.dumps({'type': 'session_title_update', 'title': title})}")
    except Exception: pass

async def execute_chat_graph_async(initial_state: dict, q: asyncio.Queue):
    user_uuid = initial_state['user_uuid']
    session_id = initial_state.get('session_id')
    
    q.put_nowait(f"__DATA__{json.dumps({'type': 'thinking_context', 'status': 'INITIALIZING_GRAPH'})}")
    
    try:
        with SessionLocal() as session:
            db_session = None
            if session_id:
                db_session = session.query(ChatSession).filter_by(session_id=session_id, user_uuid=user_uuid).first()
            
            if not db_session:
                new_session_id = session_id if session_id else str(uuid.uuid4())
                db_session = ChatSession(session_id=new_session_id, user_uuid=user_uuid, title="Analyzing...")
                session.add(db_session)
                session.commit()
                initial_state['session_id'] = new_session_id
                session_id = new_session_id
                asyncio.create_task(generate_session_title(session_id, initial_state['user_input'], q))
            else:
                db_session.last_updated = datetime.utcnow()
                session.commit()

            new_msg = ChatHistory(user_uuid=user_uuid, session_id=session_id, role="user", content=initial_state['user_input'])
            session.add(new_msg)
            session.commit()
    except Exception as e:
        logger.error(f"DB failed: {e}")

    workflow = StateGraph(BudAIState)
    workflow.add_node("ingest_context", ingest_context_node)
    workflow.add_node("intent_router", intent_router_node)
    workflow.add_node("account_clarifier", create_account_clarifier_node(q))
    workflow.add_node("analyser", analyser_worker.run_analyser_worker)
    workflow.add_node("forecaster", forecaster_worker.run_forecaster_worker)
    workflow.add_node("categorizer", categorizer_worker.run_categorizer_worker)
    workflow.add_node("health", health_worker.run_health_worker)
    workflow.add_node("ui_tagger", ui_router_node)
    workflow.add_node("market", market_worker.run_market_worker)
    workflow.add_node("response_generator", create_response_generator_node(q))
    workflow.add_node("explainer", create_explainer_node(q))
    workflow.add_node("memory", memory_worker.run_memory_worker)
    workflow.set_entry_point("ingest_context")
    workflow.add_conditional_edges("ingest_context", route_initial, {"explainer": "explainer", "intent_router": "intent_router"})
    workflow.add_conditional_edges("intent_router", route_to_worker, {
        "analyser": "analyser", 
        "forecaster": "forecaster", 
        "categorizer": "categorizer", 
        "health": "health", 
        "memory": "memory", 
        "market": "market", 
        "general": "response_generator",
        "account_clarifier": "account_clarifier"
    })
    workflow.add_edge("analyser", "ui_tagger")
    workflow.add_edge("forecaster", "ui_tagger")
    workflow.add_edge("categorizer", "ui_tagger")
    workflow.add_edge("health", "ui_tagger")
    workflow.add_edge("memory", "ui_tagger")
    workflow.add_edge("market", "ui_tagger")
    workflow.add_edge("ui_tagger", "response_generator")
    workflow.add_edge("response_generator", END)
    workflow.add_edge("explainer", END)
    workflow.add_edge("account_clarifier", END)
    budai_app = workflow.compile()
    try:
        await budai_app.ainvoke(initial_state)
    except Exception as e:
        logger.error(f"Graph failed: {e}")
        q.put_nowait("\n\n[Internal Engine Error]")
    finally:
        q.put_nowait(None)
