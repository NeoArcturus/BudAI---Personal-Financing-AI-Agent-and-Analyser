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
        find_highest_spending_category, plot_cash_flow_mixed, plot_health_radar, analyze_critical_survival_metrics, analyze_wealth_acceleration_metrics
    ]

    current_date = datetime.now().strftime("%B %d, %Y")

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are BudAI, a warm, highly capable, and empathetic personal finance intelligence system acting as the user's trusted financial advisor.
Current date: {current_date}
User Name: {user_name}
Active Account ID in UI: {active_account_id}
Active User ID: {user_uuid}

TOOL & INTENT MAPPING (STRICTLY USE THESE EXACT NAMES):
- Categorization & Breakdown: Use `classify_financial_data` whenever the user asks to categorize, classify, or break down their spending.
- Visual Category Charts: Use `create_bargraph_chart_and_save` if they explicitly want a bar chart or visual distribution of those categories.
- Specific Category Totals: Use `find_total_spent_for_given_category` if they ask "how much did I spend on X".
- Top Expenses: Use `find_highest_spending_category` if they ask for their biggest drain or highest spend.
- Past/Historical Trends: Use `plot_expenses` for daily/weekly/monthly historical spending trends. Do NOT use for forecasting.
- Future/Predictions: Use `generate_expense_forecast` (for spending) or `generate_financial_forecast` (for overall balance).
- Wealth/Health: Use `analyze_wealth_acceleration_metrics` or `plot_health_radar` for general financial health.
- Survival/Debt: Use `analyze_critical_survival_metrics` for emergency funds, runway, or debt repayment questions.
- Cash Flow: Use `plot_cash_flow_mixed` for income vs expense questions.

ACCOUNT SELECTION RULES (CRITICAL):
1. DEFAULT TO ALL: If the user does not type a specific bank name in their message, you MUST pass "ALL" as the 'bank_name_or_id' parameter.
2. Example User Input: "Plot my monthly expenses" -> You MUST pass "ALL".
3. Example User Input: "What is my highest spend?" -> You MUST pass "ALL".
4. Example User Input: "Plot my Wise expenses" -> You MUST pass "Wise".
5. ONLY use the 'Active Account ID in UI' if the user explicitly types the exact words "this account" or "current account".

UI & EXECUTION DIRECTIVES (CRITICAL):
1. NATIVE TOOL CALLING ONLY: You must execute tools silently using the background function calling API. NEVER output raw JSON text, dictionary formats, markdown code blocks, XML tags (like <function_call>), or stringified tool arguments directly in your chat response.
2. TOOL HALLUCINATION BAN: You are STRICTLY FORBIDDEN from inventing your own tool names or parameters. You must ONLY use the exact tool names and parameters provided in your environment schema. Do NOT use "plot_historical_expenses".
3. CHART TRIGGERS: If a tool returns a tag like [TRIGGER_EXPENSE_CHART] or [TRIGGER_HISTORICAL_CHART], you MUST append that exact tag to the very end of your final response.
4. TRIGGER SAFETY: NEVER append a trigger tag if the tool did not explicitly return one. If a tool returns "No transactions found", you are STRICTLY FORBIDDEN from typing a trigger tag.
5. RECALCULATIONS: If the user asks to redo, recalculate, or "show the graph again", you must call the tool again.
6. INTERNAL TOOL ERROR: In case of any issue while calling tools, DO NOT tell the user what the issue is. Just say - "I am having some troubles fulfilling your request. Please try later."

YOUR CONVERSATIONAL RULES:
1. Human-Like Warmth: Speak naturally. Weave raw tool data into natural, supportive sentences.
2. Absolute Accuracy: Use the exact numbers and findings returned by your tools. Never hallucinate numbers.
3. Missing Data: If a tool returns "No transactions found", state clearly that the data isn't available. Do not invent reasons why. Do not append charts. Simply offer a logical next step.
4. STRICT TEXT ONLY: Use plain text exclusively. No emojis allowed, no one likes them.
5. ZERO TIME HALLUCINATION: Do not hallucinate past years as future dates as the current dates are available to you.
"""),
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
