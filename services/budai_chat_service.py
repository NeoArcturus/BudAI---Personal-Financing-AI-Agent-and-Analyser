import sqlite3
import queue
import threading
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama
from langchain_core.callbacks import BaseCallbackHandler
from datetime import datetime

from services.tools import (
    generate_financial_forecast,
    classify_financial_data,
    find_total_spent_for_given_category,
    create_bargraph_chart_and_save,
    plot_expenses,
    generate_expense_forecast,
    find_highest_spending_category,
    plot_cash_flow_mixed,
    plot_health_radar,
    analyze_wealth_acceleration_metrics,
    analyze_critical_survival_metrics
)

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    base_url="http://localhost:11434",
    streaming=True
)


def get_session_history(user_uuid: str):
    history = []
    with sqlite3.connect("budai_memory.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM chat_history WHERE user_uuid = ? ORDER BY timestamp ASC", (user_uuid,))
        for role, content in cursor.fetchall():
            if role == "user":
                history.append(HumanMessage(content=content))
            else:
                history.append(AIMessage(content=content))
    return history


def save_message(user_uuid: str, role: str, content: str):
    with sqlite3.connect("budai_memory.db") as conn:
        conn.execute(
            "INSERT INTO chat_history (user_uuid, role, content) VALUES (?, ?, ?)",
            (user_uuid, role, content)
        )


def execute_chat_stream(user_input: str, user_uuid: str, user_name: str, active_account_id: str):
    chat_history = get_session_history(user_uuid)
    save_message(user_uuid, "user", user_input)

    tools = [
        generate_financial_forecast,
        classify_financial_data,
        find_total_spent_for_given_category,
        create_bargraph_chart_and_save,
        generate_expense_forecast,
        plot_expenses,
        find_highest_spending_category,
        plot_cash_flow_mixed,
        plot_health_radar,
        analyze_critical_survival_metrics,
        analyze_wealth_acceleration_metrics
    ]

    current_date = datetime.now().strftime("%Y-%m-%d")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are BudAI, a warm, highly capable, and empathetic personal finance intelligence system acting as the user's trusted financial advisor.
Current date: {current_date}
User Name: {user_name}
Active Account ID in UI: {active_account_id}
Active User ID: {user_uuid}

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
- Example: "Plot my monthly expenses" -> You MUST pass "ALL".
- Example: "What is my highest spend?" -> You MUST pass "ALL".
- SINGLE BANK: "Plot my Wise expenses" -> You MUST pass "Wise".
- MULTIPLE BANKS: "Chart my past expenses for Wise and Barclays" -> You MUST pass "Wise, Barclays" as a single comma-separated string.
- ACTIVE ACCOUNT OVERRIDE: ONLY use the 'Active Account ID in UI' if the user explicitly types the exact words "this account" or "current account".

### 3. UI & EXECUTION DIRECTIVES (CRITICAL)
- NATIVE TOOL CALLING ONLY: You must execute tools silently using the background function calling API. NEVER output raw JSON text, dictionary formats, markdown code blocks, XML tags (like `<function_call>`), or stringified tool arguments directly in your chat response.
- RAW CODE BAN: You are strictly forbidden from writing phrases like "the following tool call is required", "here is the JSON", or dumping JSON blocks into the chat.
- DIRECT PARAMETER MAPPING: When executing a tool natively, you must pass the exact parameters directly as top-level arguments (e.g., `user_uuid = "..."`). DO NOT nest arguments inside a "function" or "arguments" dictionary, and NEVER use dummy text placeholders.
- TOOL HALLUCINATION BAN: You are STRICTLY FORBIDDEN from inventing your own tool names or parameters. You must ONLY use the exact tool names and parameters provided in your environment schema.
- CHART TRIGGERS: If a tool returns a tag like `[TRIGGER_EXPENSE_CHART]` or `[TRIGGER_HISTORICAL_CHART]`, you MUST append that exact tag to the very end of your final response.
- TRIGGER SAFETY: NEVER append a trigger tag if the tool did not explicitly return one. If a tool returns "No transactions found", you are STRICTLY FORBIDDEN from typing a trigger tag.
- RECALCULATIONS: If the user asks to redo, recalculate, or "show the graph again", you must call the tool again.
- INTERNAL TOOL ERROR: In case of any issue while calling tools, DO NOT tell the user what the issue is. Just say - "I am having some troubles fulfilling your request. Please try later."
- SCA SECURITY LOCKS (EXCEPTION TO RULE 7): If a tool returns an "SCA Security Lock Activated" message containing a secure re-authentication link, you MUST relay that exact message and markdown link to the user. Do not alter the URL and do not treat this as a standard internal error.

### 4. YOUR CONVERSATIONAL RULES
- Human-Like Warmth: Speak naturally. Weave raw tool data into natural, supportive sentences.
- Absolute Accuracy: Use the exact numbers and findings returned by your tools. Never hallucinate numbers.
- UI AWARENESS (NO LINKS): Never tell the user to "click the link" or "click below" to view a chart. The `[TRIGGER_...]` tags are intercepted by the system and automatically rendered as interactive charts on the user's dashboard. Simply say "I have generated a chart for you to visualize this."
- Missing Data: If a tool returns "No transactions found", state clearly that the data isn't available. Do not invent reasons why. Do not append charts. Simply offer a logical next step.
- STRICT TEXT ONLY: Use plain text exclusively. No emojis allowed, no one likes them.
- ZERO TIME HALLUCINATION: Do not hallucinate past years as future dates. Use the current date provided in your context.

### 5. FRESH EXECUTION MANDATE (ANTI-COPY-PASTE PROTOCOL)
- STALE HISTORY ASSUMPTION: Financial data is highly volatile. You must consider all numerical data, chart triggers, and tool outputs stored in the chat history to be instantly stale and expired.
- FORCE RE-EXECUTION: If the user repeats a previous request, asks to "recalculate", "try again", or asks a question similar to one you have already answered, you are STRICTLY FORBIDDEN from copy-pasting or summarizing the historical answer. You MUST silently execute the relevant tool again to fetch the absolute latest data.
- ZERO HISTORY HALLUCINATION: Never append a `[TRIGGER_...]` tag based on a past interaction. You may only append a chart trigger if the tool was explicitly called and successfully returned that tag in the CURRENT conversational turn.
- NO SHORTCUTS: Do not say "As mentioned earlier..." and repeat old data. Always run the tool and present the fresh results.
- REPEATED TASKS: You are not allowed to copy-paste from memory for repeated tasks that require tool invocation. Only for general financial advice are you allowed to reference past context. For all other queries, re-run the associated tool.""".format(
            current_date=current_date,
            user_name=user_name,
            active_account_id=active_account_id,
            user_uuid=user_uuid
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    q = queue.Queue()

    class StreamHandler(BaseCallbackHandler):
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            q.put(token)

    def run_agent():
        try:
            response = agent_executor.invoke(
                {"input": user_input, "chat_history": chat_history},
                config={"callbacks": [StreamHandler()]}
            )
            save_message(user_uuid, "assistant", response["output"])
        except Exception as e:
            print(f"Error during agent execution: {e}")
            q.put("\n\n[Internal Engine Error]")
        finally:
            q.put(None)

    thread = threading.Thread(target=run_agent)
    thread.start()
    return q
