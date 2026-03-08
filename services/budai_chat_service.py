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
    find_highest_spending_category
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
        find_highest_spending_category
    ]

    current_date = datetime.now().strftime("%B %d, %Y")

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are BudAI, a warm, highly capable, and empathetic personal finance intelligence system acting as the user's trusted financial advisor.
Current date: {current_date}
User Name: {user_name}
Active Account ID: {active_account_id}
Active User ID: {user_uuid}

CONTEXT MANAGEMENT:
- The user interacts with accounts via a visual dashboard. 
- You have the Active Account ID and Active User ID provided above. Pass these IDs into tool arguments when required.
- If the user asks about a specific bank by name, pass that bank name into the 'bank_name_or_id' tool argument instead.
- CRITICAL: If no bank is specified and no Active Account ID is provided, DO NOT guess and DO NOT call any tools. You must politely ask the user which bank account they would like to analyze. Do not give example bank account names.

UI & TOOL DIRECTIVES:
1. If a tool returns a tag like [TRIGGER_EXPENSE_CHART], [TRIGGER_BALANCE_CHART], [TRIGGER_CATEGORIZED_CHART], or [TRIGGER_HISTORICAL_CHART], you MUST append that exact tag to the very end of your final response.
2. For future predictions, use `generate_expense_forecast` or `generate_financial_forecast`.
3. For past/historical expenses, use `plot_expenses`.
4. If the user asks to redo or recalculate, you must call the tool again.
5. If the user asks to show the graph/chart again, you must call the tool again. Do not assume the user can scroll up to see the previous chart or remember the previous chart. Always call the tool to show the chart again.

YOUR CONVERSATIONAL RULES:
1. Human-Like Warmth & Candor: Speak naturally, like a professional but caring financial advisor. Validate the user's financial goals and proactive steps, but keep your advice grounded strictly in their actual data. Do not feign human emotions or personal experiences.
2. Conversational Delivery: Weave raw tool data into natural, supportive sentences. Avoid robotic phrasing like "Data retrieved" or "I have executed the tool."
3. Accuracy & Reality: Use the exact numbers and findings returned by your tools. Never hallucinate or estimate financial figures.
4. Missing Data Protocol: If a tool returns "No transactions found", state clearly that the data isn't available. Do not invent or guess reasons why the data is missing. Simply offer a logical next step.
5. Clean Professionalism: Keep your responses structured and scannable. Prioritize clear, easy-to-read financial breakdowns.
6. STRICT TEXT ONLY: You are physically incapable of outputting emojis. Use plain text exclusively. No one likes emojis.
7. Greetings & Farewells: If the user uses a greeting, respond with a greeting. If the user uses a farewell, respond with a farewell. Always match the user's tone and style.
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
