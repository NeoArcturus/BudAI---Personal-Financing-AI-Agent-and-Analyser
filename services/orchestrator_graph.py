import threading
from datetime import datetime
import re
import json
import sys
import langchain
import uuid
from models.database_models import ChatHistory, ChatSession
from config import SessionLocal
from services.workers import analyser_worker, forecaster_worker, categorizer_worker, health_worker, memory_worker, market_worker
from models.graph_state import BudAIState
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler, StdOutCallbackHandler
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import Literal, Optional
import asyncio
import queue
import time
from services.mcp_bridge import MCPBridge
from services.mcp_tools.external_tools import export_advisory_state
from services.logger_setup import get_core_logger

logger = get_core_logger("orchestrator_graph")

langchain.debug = False


class RouteDecision(BaseModel):
    reasoning: str = Field(
        ..., description="Step-by-step reasoning explaining why this worker was selected.")
    selected_worker: Literal["analyser", "forecaster",
                             "categorizer", "health", "memory", "market", "general"] = Field(...)


router_llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    keep_alive=300
)

current_date = datetime.now().strftime("%Y-%m-%d")
current_year = datetime.now().year


def strip_thinking(text: str) -> str:
    logger.debug("Stripping thinking tags from text")
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_session_history(user_uuid: str, session_id: Optional[str] = None):
    logger.info(f"Fetching session history for user {user_uuid}, session {session_id}")
    history = []
    try:
        with SessionLocal() as session:
            query = session.query(ChatHistory).filter(ChatHistory.user_uuid == user_uuid)
            if session_id:
                query = query.filter(ChatHistory.session_id == session_id)
            else:
                query = query.filter(ChatHistory.session_id == None)
                
            records = query.order_by(ChatHistory.timestamp.asc()).all()
            logger.debug(f"Found {len(records)} history records")
            for r in records:
                if r.role == "user":
                    history.append(HumanMessage(content=r.content))
                else:
                    history.append(AIMessage(content=r.content))
    except Exception as e:
        logger.error(f"Failed to fetch session history: {e}")
    return history


def ingest_context_node(state: BudAIState):
    bridge = MCPBridge()
    try:
        local_rules = bridge.read_user_rules()
        logger.debug("Local rules ingested successfully")
    except Exception as e:
        logger.error(f"Failed to ingest local rules: {e}")
        local_rules = {}
    return {"local_rules": local_rules}


def intent_router_node(state: BudAIState):
    chat_history = get_session_history(state['user_uuid'], state.get('session_id'))

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an INVISIBLE, ROBOTIC query routing mechanism for a financial backend. YOU ARE NOT AN AI ASSISTANT. YOU ARE NOT BUDAI.

         CRITICAL DIRECTIVES:
         1. DO NOT greet the user.
         2. DO NOT answer the user's question.
         3. DO NOT hallucinate, invent, or calculate financial data.
         4. Ignore the conversational tone of any previous chat history. Your ONLY purpose is to classify the user's latest query.

         CONTEXT:
         Current Year: {current_year}

         ROUTING LOGIC:
         - Breakdown/categories ("categorize", "category", "classify", "breakdown") -> categorizer
         - Past/historical data or Exports/Custom statements (Year <= {current_year}, "spent", "history", "statement", "export", "download", "cash flow", "income vs expense") -> analyser
         - Future predictions (Year > {current_year}, "forecast", "predict", "next month") -> forecaster
         - Survival/Debt/Health ("score", "runway", "debt") -> health
         - Memory/Preferences ("remember", "forget", "my preference", "what do you know") -> memory
         - Economy/Markets/News/FX ("economy", "stock", "market", "inflation", "currency", "exchange rate", "buy abroad") -> market
         - Non-data conversational questions -> general

         Think step-by-step and provide detailed reasoning in the 'reasoning' field before outputting the selected worker.
         """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Classify this query: {input}")
    ])

    structured_router_llm = router_llm.with_structured_output(RouteDecision)
    chain = prompt | structured_router_llm

    logger.debug("Invoking Router LLM")
    try:
        result = chain.invoke({
            "input": state['user_input'],
            "current_year": current_year,
            "chat_history": chat_history
        })
        decision = result.selected_worker
        logger.info(f"Router Decision: {decision.upper()}")
        logger.debug(f"Router Reasoning: {result.reasoning}")
    except Exception as e:
        logger.error(f"LLM Routing Call Failed: {type(e).__name__} - {e}")
        logger.info("Defaulting to 'general' route")
        decision = "general"

    return {"selected_worker": decision}


def route_to_worker(state: BudAIState):
    worker = state["selected_worker"]
    logger.info(f"Routing to worker: {worker}")
    return worker


def ui_router_node(state: BudAIState):
    cache_id = state.get('cache_id')
    chart_type = state.get('chart_type')
    logger.debug(f"Cache ID: {cache_id} | Chart Type: {chart_type}")
    
    if not cache_id or not chart_type:
        logger.info("No UI trigger required")
        return {"ui_trigger_tag": ""}
        
    constructed_tag = f"[TRIGGER_{chart_type}:{cache_id}]"
    logger.info(f"Constructed UI Tag: {constructed_tag}")
    return {"ui_trigger_tag": constructed_tag}


def create_response_generator_node(q: queue.Queue):
    def response_generator_node(state: BudAIState):
        
        class StreamHandler(BaseCallbackHandler):
            def __init__(self):
                self.is_thinking = False
                self.buffer = ""
                self.lock = threading.Lock()

            def on_llm_new_token(self, token: str, **kwargs) -> None:
                with self.lock:
                    self.buffer += token
                    if self.is_thinking:
                        if "</think>" in self.buffer:
                            self.is_thinking = False
                            parts = self.buffer.split("</think>", 1)
                            self.buffer = parts[1].lstrip()
                    else:
                        if "<think>" in self.buffer:
                            self.is_thinking = True
                            parts = self.buffer.split("</think>", 1)
                            before_think = parts[0].rstrip(
                            ) if parts[0] else ""
                            if before_think:
                                q.put(before_think)
                            self.buffer = parts[1] if len(parts) > 1 else ""
                        elif "<" in self.buffer:
                            last_open = self.buffer.rfind("<")
                            if last_open > 0:
                                q.put(self.buffer[:last_open])
                                self.buffer = self.buffer[last_open:]
                            else:
                                q.put(self.buffer)
                                self.buffer = ""
                        else:
                            q.put(self.buffer)
                            self.buffer = ""

            def finalize(self):
                if not self.is_thinking and self.buffer:
                    q.put(self.buffer)
                    self.buffer = ""

        stream_handler = StreamHandler()

        persona_llm = ChatOllama(
            model="qwen3:4b",
            temperature=0.4,
            keep_alive=300
        )

        chat_history = get_session_history(state['user_uuid'], state.get('session_id'))

        from services.profile_builder import ProfileBuilder
        profile_builder = ProfileBuilder(state['user_uuid'])
        mrfp = profile_builder.build_profile()

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are BudAI, a precise senior financial advisor.
             
        CRITICAL OPERATIONAL MANDATES:
        1. ZERO HALLUCINATION: Never invent, estimate, or assume financial figures. Only use mathematically verified data provided in the USER PROFILE below.
        2. NO EMOJIS: Use plain text only. Emojis are strictly forbidden.
        3. GBP ONLY: All currency must be in GBP (£).
        4. VERBATIM TRIGGERS: If a tool returns a `[TRIGGER_...:CACHE_...]` tag, you must include it as the final line of your response.

        ### USER PROFILE
        {mrfp}

        ### YOUR ROLE
        Provide warm, technically accurate, and empathetic financial advice based on the user's profile and the data returned by tools.
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "User: {input}\nData Summary: {worker_summary}")
        ])

        chain = (prompt | persona_llm).with_config({
            "callbacks": [stream_handler, StdOutCallbackHandler()],
            "verbose": True
        })

        logger.info("Invoking Persona LLM for final response")
        try:
            result = chain.invoke({
                "input": state['user_input'],
                "worker_summary": state.get('worker_summary', 'No specific data found.'),
                "chat_history": chat_history
            })
        except Exception as e:
            logger.error(f"Persona LLM Call Failed: {type(e).__name__} - {e}")
            raise e

        stream_handler.finalize()
        warm_text = strip_thinking(result.content)
        ui_tag = state.get("ui_trigger_tag", "")

        if ui_tag:
            final_output = f"{warm_text}\n\n{ui_tag}".strip()
            clean_content = final_output.replace(ui_tag, "").strip()
        else:
            final_output = warm_text.strip()
            clean_content = final_output.strip()

        logger.debug("Storing response in database")
        try:
            with SessionLocal() as session:
                if clean_content:
                    new_msg = ChatHistory(
                        user_uuid=state['user_uuid'],
                        session_id=state.get('session_id'),
                        role="assistant",
                        content=clean_content
                    )
                    session.add(new_msg)
                    session.commit()
                    logger.info("Assistant message stored successfully")

                    if state.get('raw_data') and state.get('chart_type'):
                        logger.info("Triggering Advisory Export")
                        try:
                            export_res = export_advisory_state.invoke({
                                "user_uuid": state['user_uuid'],
                                "chart_type": state['chart_type'],
                                "raw_data": state['raw_data'],
                                "ai_analysis": clean_content
                            })
                            logger.info(f"Advisory Export success: {export_res}")
                        except Exception as mcp_e:
                            logger.error(f"MCP Export Failed: {mcp_e}")

                else:
                    logger.info("LLM output was empty, skipping DB storage")
        except Exception as db_e:
            logger.error(f"Failed to store Assistant output in database: {db_e}")

        if ui_tag:
            logger.debug(f"Putting UI tag in queue: {ui_tag}")
            q.put(f"\n\n{ui_tag}")

        return {"final_response": final_output}
    return response_generator_node


def create_explainer_node(q: queue.Queue):
    def explainer_node(state: BudAIState):
        
        class StreamHandler(BaseCallbackHandler):
            def __init__(self):
                self.is_thinking = False
                self.buffer = ""
                self.lock = threading.Lock()

            def on_llm_new_token(self, token: str, **kwargs) -> None:
                with self.lock:
                    self.buffer += token
                    if self.is_thinking:
                        if "</think>" in self.buffer:
                            self.is_thinking = False
                            parts = self.buffer.split("</think>", 1)
                            self.buffer = parts[1].lstrip()
                    else:
                        if "<think>" in self.buffer:
                            self.is_thinking = True
                            parts = self.buffer.split("</think>", 1)
                            before_think = parts[0].rstrip(
                            ) if parts[0] else ""
                            if before_think:
                                q.put(before_think)
                            self.buffer = parts[1] if len(parts) > 1 else ""
                        elif "<" in self.buffer:
                            last_open = self.buffer.rfind("<")
                            if last_open > 0:
                                q.put(self.buffer[:last_open])
                                self.buffer = self.buffer[last_open:]
                            else:
                                q.put(self.buffer)
                                self.buffer = ""
                        else:
                            q.put(self.buffer)
                            self.buffer = ""

            def finalize(self):
                if not self.is_thinking and self.buffer:
                    q.put(self.buffer)
                    self.buffer = ""

        stream_handler = StreamHandler()

        persona_llm = ChatOllama(
            model="qwen3:4b",
            temperature=0.3,
            keep_alive=300
        )

        chat_history = get_session_history(state['user_uuid'], state.get('session_id'))
        current_date_str = datetime.now().strftime("%Y-%m-%d")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are BudAI, an expert financial explainer. 
            Current Context Date: {current_date}
            
            The user is viewing data generated by the orchestrator tool: {chart_type}.
            You are provided with the exact JSON data rendering that visualization.
            
            Always think step-by-step and wrap your internal reasoning inside <think>...</think> tags before answering.

            YOUR DIRECTIVES:
            1. Analyze the JSON data provided. Point out critical insights, trends, or anomalies.
            2. Speak directly to the user in a warm, professional tone. Keep it concise.
            3. TIME AWARENESS (CRITICAL): Compare the dates in the data to the Current Date ({current_date}). DO NOT call past dates a 'future projection' or 'forecast'. 
            4. FORECAST RULE: Only apply forecast rules if the tool name '{chart_type}' explicitly contains the word 'forecast'. If it does, DO NOT categorize expenses.
            5. CURRENCY FORMATTING: You MUST format all financial values and money amounts using the GBP (£) symbol. Never use the USD ($) symbol.
            
            6. MANDATORY NEXT ACTIONS: At the very end of your explanation, provide 2-3 valid follow-up actions the user can take. Format them strictly as a JSON array inside a markdown block.
             7. TEXTING RULES: Strict plain text only. No emojis allowed.
            
            VALID TOOL TARGETS YOU MUST CHOOSE FROM:
            - 'classify_financial_data' 
            - 'plot_health_radar' 
            - 'historical_monthly' 
            - 'expense_forecast' 
            - 'balance_forecast' 
            
            Example format to append at the end:
            ```json
            [
              {{"label": "Analyze my historical categories", "tool_target": "classify_financial_data"}},
              {{"label": "Show my overall financial health", "tool_target": "plot_health_radar"}}
            ]
            ```
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human",
             "Here is the raw data I am looking at right now: {raw_data}")
        ])

        chain = (prompt | persona_llm).with_config({
            "callbacks": [stream_handler, StdOutCallbackHandler()],
            "verbose": True
        })

        logger.info("Invoking Explainer LLM")
        try:
            result = chain.invoke({
                "chart_type": state.get('chart_type', 'UNKNOWN_CHART'),
                "raw_data": json.dumps(state.get('raw_data', {})),
                "current_date": current_date_str,
                "chat_history": chat_history
            })
        except Exception as e:
            logger.error(f"Explainer LLM Failed: {e}")
            raise e

        stream_handler.finalize()
        warm_text = strip_thinking(result.content)

        logger.debug("Storing explainer response in database")
        try:
            with SessionLocal() as session:
                new_msg = ChatHistory(
                    user_uuid=state['user_uuid'],
                    session_id=state.get('session_id'),
                    role="assistant",
                    content=warm_text
                )
                session.add(new_msg)
                session.commit()
                logger.info("Explainer message stored successfully")
        except Exception as db_e:
            logger.error(f"Failed to store Explainer Assistant output: {db_e}")

        return {"final_response": warm_text}
    return explainer_node


def route_initial(state: BudAIState):
    is_explanation = state.get("is_explanation")
    if is_explanation:
        return "explainer"
    return "intent_router"


def generate_session_title(session_id: str, first_msg: str):
    try:
        llm = ChatOllama(model="qwen3:4b", temperature=0.1)
        res = llm.invoke(f"Generate a concise 3-4 word professional title for a financial chat starting with: '{first_msg}'. Output only the title, no quotes or punctuation.")
        title = res.content.strip().replace('"', '').replace('Title: ', '')
        with SessionLocal() as session:
            db_session = session.query(ChatSession).filter_by(session_id=session_id).first()
            if db_session:
                db_session.title = title
                session.commit()
    except Exception:
        pass


def execute_chat_graph(initial_state: dict):
    q = queue.Queue()
    user_uuid = initial_state['user_uuid']
    session_id = initial_state.get('session_id')
    
    try:
        with SessionLocal() as session:
            if not session_id:
                session_id = str(uuid.uuid4())
                initial_state['session_id'] = session_id
                logger.info(f"Creating new chat session: {session_id}")
                new_session = ChatSession(
                    session_id=session_id,
                    user_uuid=user_uuid,
                    title="Analyzing Context..."
                )
                session.add(new_session)
                session.commit()
                threading.Thread(target=generate_session_title, args=(session_id, initial_state['user_input'])).start()
            else:
                logger.debug(f"Verifying session ID: {session_id}")
                existing_session = session.query(ChatSession).filter_by(session_id=session_id, user_uuid=user_uuid).first()
                if not existing_session:
                    session_id = str(uuid.uuid4())
                    initial_state['session_id'] = session_id
                    logger.info(f"Invalid session ID, created new: {session_id}")
                    new_session = ChatSession(
                        session_id=session_id,
                        user_uuid=user_uuid,
                        title=initial_state['user_input'][:50] + "..." if len(initial_state['user_input']) > 50 else initial_state['user_input']
                    )
                    session.add(new_session)
                    session.commit()
                else:
                    existing_session.last_updated = datetime.utcnow()
                    session.commit()
                    logger.debug("Session updated")

            new_msg = ChatHistory(
                user_uuid=user_uuid,
                session_id=session_id,
                role="user",
                content=initial_state['user_input']
            )
            session.add(new_msg)
            session.commit()
            logger.info("User message stored in history")
    except Exception as e:
        logger.error(f"Database operation failed in execute_chat_graph: {e}")

    logger.debug("Compiling LangGraph workflow")
    workflow = StateGraph(BudAIState)

    workflow.add_node("ingest_context", ingest_context_node)
    workflow.add_node("intent_router", intent_router_node)
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

    workflow.add_conditional_edges(
        "ingest_context",
        route_initial,
        {
            "explainer": "explainer",
            "intent_router": "intent_router"
        }
    )

    workflow.add_conditional_edges(
        "intent_router",
        route_to_worker,
        {
            "analyser": "analyser",
            "forecaster": "forecaster",
            "categorizer": "categorizer",
            "health": "health",
            "memory": "memory",
            "market": "market",
            "general": "response_generator"
        }
    )

    workflow.add_edge("analyser", "ui_tagger")
    workflow.add_edge("forecaster", "ui_tagger")
    workflow.add_edge("categorizer", "ui_tagger")
    workflow.add_edge("health", "ui_tagger")
    workflow.add_edge("memory", "ui_tagger")
    workflow.add_edge("market", "ui_tagger")
    workflow.add_edge("ui_tagger", "response_generator")
    workflow.add_edge("response_generator", END)
    workflow.add_edge("explainer", END)

    budai_app = workflow.compile()
    logger.info("Graph workflow compiled successfully")

    def run_graph():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(budai_app.ainvoke(initial_state))
        except Exception as e:
            logger.error(f"Exception inside graph thread: {e}")
            q.put("\n\n[Internal Engine Error]")
        finally:
            q.put(None)
            loop.close()

    threading.Thread(target=run_graph).start()
    return q

