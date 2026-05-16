import threading
from datetime import datetime
import re
import json
import sys
import logging
import langchain
import uuid
from models.database_models import ChatHistory, ChatSession
from config import SessionLocal
from services.workers import analyser_worker, forecaster_worker, categorizer_worker, health_worker, memory_worker
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

logger = logging.getLogger("uvicorn.error")

langchain.debug = True


class RouteDecision(BaseModel):
    reasoning: str = Field(
        ..., description="Step-by-step reasoning explaining why this worker was selected.")
    selected_worker: Literal["analyser", "forecaster",
                             "categorizer", "health", "memory", "general"] = Field(...)


router_llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    keep_alive=300
)

current_date = datetime.now().strftime("%Y-%m-%d")
current_year = datetime.now().year


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_session_history(user_uuid: str, session_id: Optional[str] = None):
    history = []
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
    return history


def ingest_context_node(state: BudAIState):
    bridge = MCPBridge()
    try:
        local_rules = bridge.read_local_rules()
    except Exception:
        local_rules = {}
    return {"local_rules": local_rules}


def intent_router_node(state: BudAIState):
    logger.info(f"Received query: {state['user_input']}")
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
         - Non-data conversational questions -> general

         Think step-by-step and provide detailed reasoning in the 'reasoning' field before outputting the selected worker.
         """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Classify this query: {input}")
    ])

    structured_router_llm = router_llm.with_structured_output(RouteDecision)
    chain = prompt | structured_router_llm

    logger.info("--- STARTING ROUTER COT ---")
    try:
        result = chain.invoke({
            "input": state['user_input'],
            "current_year": current_year,
            "chat_history": chat_history
        })
        decision = result.selected_worker
        logger.info(f"Router Reasoning: {result.reasoning}")
    except Exception as e:
        logger.error(
            f"LLM Routing Call or Validation Failed: {type(e).__name__} - {e}")
        logger.info("Defaulting to 'general' route due to validation error.")
        decision = "general"

    logger.info("--- ENDING ROUTER COT ---")
    logger.info(f"Final Routing Decision: {decision.upper()}")
    return {"selected_worker": decision}


def route_to_worker(state: BudAIState):
    return state["selected_worker"]


def ui_router_node(state: BudAIState):
    logger.info(
        f"Received cache id: {state.get('cache_id')} | Chart type: {state.get('chart_type')}")
    if not state.get("cache_id") or not state.get("chart_type"):
        return {"ui_trigger_tag": ""}
    constructed_tag = f"[TRIGGER_{state['chart_type']}:{state['cache_id']}]"
    logger.info(f"Appending Tag: {constructed_tag}")
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

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are BudAI, a warm, highly capable, and empathetic personal finance intelligence system acting as the user's trusted financial advisor.

        Always think step-by-step and wrap your internal reasoning inside <think>...</think> tags before answering.

        ### 1. TOOL & INTENT MAPPING (STRICT)
        - Categorization & Breakdown: Use `classify_financial_data` whenever the user asks to categorize, classify, or break down their spending.
        - Visual Category Charts: Use `create_bargraph_chart_and_save` if they explicitly want a bar chart or visual distribution of those categories.
        - Specific Category Totals: Use `find_total_spent_for_given_category` if they ask "how much did I spend on X".
        - Top Expenses: Use `find_highest_spending_category` if they ask for their biggest drain or highest spend.
        - Past/Historical Trends: Use `plot_expenses` for daily/weekly/monthly historical spending trends. Do NOT use for forecasting. *CRITICAL: If the user asks for historical charts but does not specify a timeframe (daily/weekly), you MUST default to "Monthly".*
        - Future/Predictions: Use `generate_expense_forecast` (for spending) or `generate_financial_forecast` (for overall balance).
        - Wealth/Health: Use `analyze_wealth_acceleration_metrics` or `plot_health_radar` for general financial health.
        - Survival/Debt: Use `analyze_critical_survival_metrics` for emergency funds, runway, or debt repayment questions.
        - Cash Flow: Use `plot_cash_flow_mixed` for income vs expense questions.

        ### 2. ACCOUNT SELECTION RULES (CRITICAL)
        - DEFAULT TO ALL: If the user does not type a specific bank name in their message, you MUST pass "ALL" as the `bank_name_or_id` parameter.
        - SINGLE BANK: "Plot my Wise expenses" -> You MUST pass "Wise".
        - MULTIPLE BANKS: "Chart my past expenses for Wise and Barclays" -> You MUST pass "Wise, Barclays" as a single comma-separated string.
        - ACTIVE ACCOUNT OVERRIDE: ONLY use the 'Active Account ID in UI' if the user explicitly types the exact words "this account" or "current account".

        ### 3. UI & EXECUTION DIRECTIVES (CRITICAL)
        - TOOL EXECUTION: You have access to tools. You MUST use them to answer data-specific questions. Do not guess or estimate financial figures.
        - DIRECT PARAMETER MAPPING: You must provide the exact parameters required by the tool.
        - TOOL HALLUCINATION BAN: You are STRICTLY FORBIDDEN from inventing your own tool names or parameters.
        - CHART TRIGGERS (ABSOLUTE MANDATE): When a tool's raw output contains a tag formatted exactly like `[TRIGGER_...:CACHE_...]`, you are structurally bound to copy that exact string and paste it as the VERY LAST LINE of your text response.
        - TRIGGER SAFETY: NEVER append a trigger tag if the tool did not explicitly return one.
        - INTERNAL TOOL ERROR: In case of any issue while calling tools, DO NOT tell the user what the issue is. Just say - "I am having some troubles fulfilling your request. Please try later."
        - SCA SECURITY LOCKS: If a tool returns an "SCA Security Lock Activated" message containing a secure re-authentication link, you MUST relay that exact message and markdown link to the user.

        ### 4. YOUR CONVERSATIONAL RULES
        - Human-Like Warmth: Speak naturally. Weave raw tool data into supportive sentences.
        - Absolute Accuracy: Use the exact numbers and findings returned by your tools. Never hallucinate numbers.
        - UI AWARENESS (NO LINKS): Never tell the user to "click the link" to view a chart. Simply say "I have generated a chart for you to visualize this."
        - Missing Data: If a tool returns "No transactions found", state clearly that the data isn't available. Do not invent reasons why.
        - STRICT TEXT ONLY: Use plain text exclusively. No emojis allowed.
        - ZERO TIME HALLUCINATION: Do not hallucinate past years as future dates. Use the current date provided in your context.
        - CURRENCY FORMATTING: You MUST format all financial values and money amounts using the GBP (£) symbol. Never use the USD ($) symbol.

        ### 5. FRESH EXECUTION MANDATE (ANTI-COPY-PASTE PROTOCOL)
        - STALE HISTORY ASSUMPTION: Financial data is highly volatile. You must consider all numerical data, chart triggers, and tool outputs stored in the chat history to be instantly stale and expired.
        - FORCE RE-EXECUTION: If the user repeats a previous request, asks to "recalculate", "explain the data", or asks a question similar to one you have already answered, you are STRICTLY FORBIDDEN from copy-pasting or summarizing the historical answer. You MUST silently execute the relevant tool again to fetch the absolute latest data and then explain it in detail.
        - ZERO HISTORY HALLUCINATION: Never append a `[TRIGGER_...]` tag based on a past interaction. You may only append a chart trigger if the tool was explicitly called and successfully returned that tag in the CURRENT conversational turn.
        - NO SHORTCUTS: Do not say "As mentioned earlier..." and repeat old data. Always run the tool and present the fresh results.
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "User: {input}\nData Summary: {worker_summary}")
        ])

        chain = (prompt | persona_llm).with_config({
            "callbacks": [stream_handler, StdOutCallbackHandler()],
            "verbose": True
        })

        logger.info("--- STARTING RESPONSE GENERATOR COT ---")
        try:
            result = chain.invoke({
                "input": state['user_input'],
                "worker_summary": state.get('worker_summary', 'No specific data found.'),
                "chat_history": chat_history
            })
        except Exception as e:
            logger.error(f"Persona LLM Call Failed: {type(e).__name__} - {e}")
            raise e
        logger.info("--- ENDING RESPONSE GENERATOR COT ---")

        stream_handler.finalize()
        warm_text = strip_thinking(result.content)
        ui_tag = state.get("ui_trigger_tag", "")

        if ui_tag:
            final_output = f"{warm_text}\n\n{ui_tag}".strip()
            clean_content = final_output.replace(ui_tag, "").strip()
        else:
            final_output = warm_text.strip()
            clean_content = final_output.strip()

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
                    logger.info("LLM output stored in database.")

                    if state.get('raw_data') and state.get('chart_type'):
                        try:
                            export_res = export_advisory_state.invoke({
                                "user_uuid": state['user_uuid'],
                                "chart_type": state['chart_type'],
                                "raw_data": state['raw_data'],
                                "ai_analysis": clean_content
                            })
                            logger.info(f"Advisory Export: {export_res}")
                        except Exception as mcp_e:
                            logger.error(f"MCP Export Failed: {mcp_e}")

                else:
                    logger.info("LLM output was empty after cleaning.")
        except Exception as db_e:
            logger.error(f"Failed to store LLM output in database: {db_e}")

        if ui_tag:
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
        current_date = datetime.now().strftime("%Y-%m-%d")

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

        logger.info("--- STARTING EXPLAINER COT ---")
        try:
            result = chain.invoke({
                "chart_type": state.get('chart_type', 'UNKNOWN_CHART'),
                "raw_data": json.dumps(state.get('raw_data', {})),
                "current_date": current_date,
                "chat_history": chat_history
            })
        except Exception as e:
            logger.error(f"Explainer LLM Failed: {e}")
            raise e
        logger.info("--- ENDING EXPLAINER COT ---")

        stream_handler.finalize()
        warm_text = strip_thinking(result.content)

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
        except Exception as db_e:
            logger.error(f"Failed to store Explainer LLM output: {db_e}")

        return {"final_response": warm_text}
    return explainer_node


def route_initial(state: BudAIState):
    if state.get("is_explanation"):
        return "explainer"
    return "intent_router"


def execute_chat_graph(initial_state: dict):
    q = queue.Queue()
    user_uuid = initial_state['user_uuid']
    session_id = initial_state.get('session_id')
    
    with SessionLocal() as session:
        if not session_id:
            # Create new session
            session_id = str(uuid.uuid4())
            initial_state['session_id'] = session_id
            new_session = ChatSession(
                session_id=session_id,
                user_uuid=user_uuid,
                title=initial_state['user_input'][:50] + "..." if len(initial_state['user_input']) > 50 else initial_state['user_input']
            )
            session.add(new_session)
            session.commit()
            logger.info(f"Created new chat session: {session_id}")
        else:
            # Verify session exists
            existing_session = session.query(ChatSession).filter_by(session_id=session_id, user_uuid=user_uuid).first()
            if not existing_session:
                # Fallback to new session if invalid ID provided
                session_id = str(uuid.uuid4())
                initial_state['session_id'] = session_id
                new_session = ChatSession(
                    session_id=session_id,
                    user_uuid=user_uuid,
                    title=initial_state['user_input'][:50] + "..." if len(initial_state['user_input']) > 50 else initial_state['user_input']
                )
                session.add(new_session)
                session.commit()
                logger.info(f"Invalid session ID provided. Created new session: {session_id}")
            else:
                existing_session.last_updated = datetime.utcnow()
                session.commit()

        new_msg = ChatHistory(
            user_uuid=user_uuid,
            session_id=session_id,
            role="user",
            content=initial_state['user_input']
        )
        session.add(new_msg)
        session.commit()

    workflow = StateGraph(BudAIState)

    workflow.add_node("ingest_context", ingest_context_node)
    workflow.add_node("intent_router", intent_router_node)
    workflow.add_node("analyser", analyser_worker.run_analyser_worker)
    workflow.add_node("forecaster", forecaster_worker.run_forecaster_worker)
    workflow.add_node("categorizer", categorizer_worker.run_categorizer_worker)
    workflow.add_node("health", health_worker.run_health_worker)
    workflow.add_node("ui_tagger", ui_router_node)
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
            "general": "response_generator"
        }
    )

    workflow.add_edge("analyser", "ui_tagger")
    workflow.add_edge("forecaster", "ui_tagger")
    workflow.add_edge("categorizer", "ui_tagger")
    workflow.add_edge("health", "ui_tagger")
    workflow.add_edge("memory", "ui_tagger")
    workflow.add_edge("ui_tagger", "response_generator")
    workflow.add_edge("response_generator", END)
    workflow.add_edge("explainer", END)

    budai_app = workflow.compile()

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
