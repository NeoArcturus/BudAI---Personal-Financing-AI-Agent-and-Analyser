import threading
from datetime import datetime
import re
import langchain
from models.database_models import ChatHistory
from config import SessionLocal
from services.workers import analyser_worker, forecaster_worker, categorizer_worker, health_worker
from models.graph_state import BudAIState
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler, StdOutCallbackHandler
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
import asyncio
import queue
import time

langchain.debug = True

router_llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    keep_alive=300,
    streaming=True,
)

current_date = datetime.now().strftime("%Y-%m-%d")
current_year = datetime.now().year


class ThinkingStreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_session_history(user_uuid: str):
    history = []
    with SessionLocal() as session:
        records = session.query(ChatHistory).filter(
            ChatHistory.user_uuid == user_uuid).order_by(ChatHistory.timestamp.asc()).all()
        for r in records:
            if r.role == "user":
                history.append(HumanMessage(content=r.content))
            else:
                history.append(AIMessage(content=r.content))
    return history


def intent_router_node(state: BudAIState):
    print(f"\n[ORCHESTRATOR] Received query: {state['user_input']}")

    chat_history = get_session_history(state['user_uuid'])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query router for a financial system.

         CRITICAL CONTEXT:
         Today's exact date is: {current_date}
         The current year is: {current_year}

         STEPS:
         1. THINK BEFORE ANSWERING: You must analyze the user's query and take a decision as to which route to go.
         2. TIME CONTEXT (MANDATORY): If the user mentions a specific year, you MUST compare it mathematically to the current year ({current_year}).
            - If the requested year is LESS THAN OR EQUAL TO {current_year}, it is PAST/HISTORICAL data -> route to 'analyser'.
            - If the requested year is GREATER THAN {current_year}, it is FUTURE data -> route to 'forecaster'.
         3. ALLOWED DECISIONS: 
            - Are they asking about past/historical data? -> analyser
            - Are they asking about future predictions or forecasts? -> forecaster
            - Are they asking to categorize or break down spending? -> categorizer
            - Are they asking about financial health, debt, or survival metrics? -> health
            - Are they asking a general finance question with no data needed? -> general
         4. CHECK FOR AMBIGUITY:
            - If the query mentions "forecast", "predict", "next X days" -> forecaster
            - If the query mentions "category", "breakdown", "classify" -> categorizer
            - If the query mentions "health", "score", "runway", "debt" -> health
            - If the query mentions "spent", "expenses", "cash flow", "history", or a past year -> analyser
            - If none of the above apply -> general
         
         5. FRESH EXECUTION MANDATE (ANTI-COPY-PASTE PROTOCOL)
            - STALE HISTORY ASSUMPTION: Financial data is highly volatile. You must consider all numerical data, chart triggers, and tool outputs stored in the chat history to be instantly stale and expired.
            - FORCE RE-EXECUTION: If the user repeats a previous request, asks to "recalculate", "explain the data", or asks a question similar to one you have already answered, you are STRICTLY FORBIDDEN from copy-pasting or summarizing the historical answer. You MUST silently execute the relevant tool again to fetch the absolute latest data and then explain it in detail.
            - ZERO HISTORY HALLUCINATION: Never append a `[TRIGGER_...]` tag based on a past interaction. You may only append a chart trigger if the tool was explicitly called and successfully returned that tag in the CURRENT conversational turn.
            - NO SHORTCUTS: Do not say "As mentioned earlier..." and repeat old data. Always run the tool and present the fresh results.
            
         6. FINAL ANSWER WRAPPER: You MUST wrap your final chosen route strictly in XML tags at the very end of your response. Example: <route>analyser</route>
         """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "User Query: {input}")
    ])

    chain = (prompt | router_llm).with_config({
        "callbacks": [ThinkingStreamHandler()],
        "verbose": True
    })

    print("\n[DEBUG] --- STARTING ROUTER COT ---")

    try:
        result = chain.invoke({
            "input": state['user_input'],
            "current_date": current_date,
            "current_year": current_year,
            "chat_history": chat_history
        })
    except Exception as e:
        print(f"[ERROR] LLM Call Failed: {type(e).__name__} - {e}")
        raise e

    print("\n[DEBUG] --- ENDING ROUTER COT ---\n")

    raw_output = result.content.strip()
    cleaned_output = strip_thinking(raw_output)

    decision = "general"
    valid = ["analyser", "forecaster", "categorizer", "health", "general"]

    match = re.search(r"<route>(.*?)</route>", cleaned_output, re.IGNORECASE)
    if match:
        extracted = match.group(1).strip().lower()
        if extracted in valid:
            decision = extracted
    else:
        lines = [l.strip().lower()
                 for l in cleaned_output.splitlines() if l.strip()]
        for line in reversed(lines):
            words = re.findall(r'[a-z]+', line)
            for word in reversed(words):
                if word in valid:
                    decision = word
                    break
            if decision != "general":
                break

    print(f"\n[ORCHESTRATOR] Final Routing Decision: {decision.upper()}")
    return {"selected_worker": decision}


def route_to_worker(state: BudAIState):
    return state["selected_worker"]


def ui_router_node(state: BudAIState):
    print(
        f"\n[UI ROUTER] Received cache id: {state.get('cache_id')} | Chart type: {state.get('chart_type')}")

    if not state.get("cache_id") or not state.get("chart_type"):
        return {"ui_trigger_tag": ""}

    constructed_tag = f"[TRIGGER_{state['chart_type']}:{state['cache_id']}]"

    print(f"[UI ROUTER] Appending Tag: {constructed_tag}")

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
            keep_alive=300,
            extra_body={"think": False},
            callbacks=[stream_handler]
        )

        chat_history = get_session_history(state['user_uuid'])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are BudAI, a warm, highly capable, and empathetic personal finance intelligence system acting as the user's trusted financial advisor.

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
        - DIRECT PARAMETER MAPPING: You must provide the exact parameters required by the tool (e.g., `user_uuid`, `bank_name_or_id`). Adhere strictly to the native tool-calling format without omitting required fields. Do not use dummy text placeholders.
        - TOOL HALLUCINATION BAN: You are STRICTLY FORBIDDEN from inventing your own tool names or parameters.
        - CHART TRIGGERS (ABSOLUTE MANDATE): When a tool's raw output contains a tag formatted exactly like `[TRIGGER_...:CACHE_...]`, you are structurally bound to copy that exact string and paste it as the VERY LAST LINE of your text response.
        * DO NOT modify the tag name.
        * DO NOT add, remove, or guess date parameters inside the brackets.
        * DO NOT wrap the tag in backticks or markdown code blocks.
        * DO NOT tell the user "I have generated a chart" if you failed to append the tag.
        * The tag must be on its own line, at the very bottom of your message.
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
            "callbacks": [stream_handler],
            "verbose": True
        })

        try:
            result = chain.invoke({
                "input": state['user_input'],
                "worker_summary": state.get('worker_summary', 'No specific data found.'),
                "chat_history": chat_history
            })
        except Exception as e:
            print(f"[ERROR] Persona LLM Call Failed: {type(e).__name__} - {e}")
            raise e

        # FIX 4: Finalize any remaining buffer
        stream_handler.finalize()

        warm_text = strip_thinking(result.content)
        ui_tag = state.get("ui_trigger_tag", "")  # Can be None

        # FIX 1: Handle None ui_tag safely
        if ui_tag:
            final_output = f"{warm_text}\n\n{ui_tag}".strip()
        else:
            final_output = warm_text.strip()

        # FIX 2: Handle None ui_tag in replace()
        if ui_tag:
            clean_content = final_output.replace(ui_tag, "").strip()
        else:
            clean_content = final_output.strip()

        try:
            with SessionLocal() as session:
                if clean_content:
                    new_msg = ChatHistory(
                        user_uuid=state['user_uuid'],
                        role="assistant",
                        content=clean_content
                    )
                    session.add(new_msg)
                    session.commit()
                    print("\n[DEBUG] LLM output stored in database.")
                else:
                    print("[DEBUG] LLM output was empty after cleaning.")
        except Exception as db_e:
            print(f"[ERROR] Failed to store LLM output in database: {db_e}")
            # FIX 5: Retry with exponential backoff
            for attempt in range(3):
                try:
                    with SessionLocal() as session:
                        session.rollback()
                        # FIX 3: Ensure clean_content is always defined
                        safe_clean_content = clean_content if 'clean_content' in locals() else final_output.strip()
                        new_msg = ChatHistory(
                            user_uuid=state['user_uuid'],
                            role="assistant",
                            content=safe_clean_content
                        )
                        session.add(new_msg)
                        session.commit()
                        print(
                            f"[RETRY] Database commit successful on attempt {attempt + 1}")
                        break
                except Exception as retry_e:
                    if attempt == 2:
                        print(f"[RETRY ERROR] All attempts failed: {retry_e}")
                    time.sleep(2 ** attempt)

        if ui_tag:
            q.put(f"\n\n{ui_tag}")

        return {"final_response": final_output}
    return response_generator_node


def execute_chat_graph(initial_state: dict):
    q = queue.Queue()
    with SessionLocal() as session:
        new_msg = ChatHistory(
            user_uuid=initial_state['user_uuid'],
            role="user",
            content=initial_state['user_input']
        )
        session.add(new_msg)
        session.commit()

    workflow = StateGraph(BudAIState)
    workflow.add_node("intent_router", intent_router_node)
    workflow.add_node("analyser", analyser_worker.run_analyser_worker)
    workflow.add_node("forecaster", forecaster_worker.run_forecaster_worker)
    workflow.add_node("categorizer", categorizer_worker.run_categorizer_worker)
    workflow.add_node("health", health_worker.run_health_worker)
    workflow.add_node("ui_tagger", ui_router_node)
    workflow.add_node("response_generator", create_response_generator_node(q))

    workflow.set_entry_point("intent_router")

    workflow.add_conditional_edges(
        "intent_router",
        route_to_worker,
        {
            "analyser": "analyser",
            "forecaster": "forecaster",
            "categorizer": "categorizer",
            "health": "health",
            "general": "response_generator"
        }
    )

    workflow.add_edge("analyser", "ui_tagger")
    workflow.add_edge("forecaster", "ui_tagger")
    workflow.add_edge("categorizer", "ui_tagger")
    workflow.add_edge("health", "ui_tagger")
    workflow.add_edge("ui_tagger", "response_generator")
    workflow.add_edge("response_generator", END)

    budai_app = workflow.compile()

    def run_graph():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            budai_app.invoke(initial_state)
        except Exception as e:
            print(f"[DEBUG ERROR] Exception inside graph thread: {e}")
            q.put("\n\n[Internal Engine Error]")
        finally:
            q.put(None)
            loop.close()

    threading.Thread(target=run_graph).start()
    return q
