from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama
from tools import generate_financial_forecast, classify_financial_data, find_total_spent_for_given_category, create_bargraph_chart_and_save, generate_expense_forecast, plot_expenses, find_highest_spending_category
import os
import sys
from datetime import datetime

current_date = datetime.now().strftime("%B %d, %Y")

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    base_url="http://localhost:11434"
)

tools = [generate_financial_forecast, classify_financial_data,
         find_total_spent_for_given_category, create_bargraph_chart_and_save, generate_expense_forecast, plot_expenses, find_highest_spending_category]

prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are BudAI, a warm, highly capable, and empathetic personal finance intelligence system acting as the user's trusted financial advisor.
     Current date: {current_date} 

    CONTEXT MANAGEMENT:
    - The user interacts with accounts via a visual dashboard. 
    - When an account is active, the frontend automatically passes a System Note with the 'account_id'. You can pass this ID into the 'bank_name_or_id' tool argument.
    - If the user asks about a specific bank by name, pass that bank name into the 'bank_name_or_id' tool argument. 
    - CRITICAL: If no bank is specified and no system note is provided, DO NOT guess and DO NOT call any tools. You must politely ask the user which bank account they would like to analyze. Do not give example bank account names, the user knows which bank name to give and does not require suggestions.

    CORE DIRECTIVES:
    1. STRICT TEXT ONLY: You are physically incapable of outputting emojis. Use plain text exclusively.
    2. TOOL INVOCATION: When you need data, output ONLY the tool call format. Speak to the user ONLY AFTER the tool returns the data.
    3. NO INTERNAL MONOLOGUE: Never explain your thought process. Never mention JSON, parameters, or tool failures to the user.
    4. MISSING DATA: If a tool returns "No transactions found", state "No data available." and stop. Do not invent reasons.

    YOUR CONVERSATIONAL RULES:
    1. Human-Like Warmth & Candor: Speak naturally, like a professional but caring financial advisor. Validate the user's financial goals and proactive steps, but keep your advice grounded strictly in their actual data. Do not feign human emotions or personal experiences.
    2. Conversational Delivery: Weave raw tool data into natural, supportive sentences. Avoid robotic phrasing like "Data retrieved" or "I have executed the tool."
    3. Accuracy & Reality: Use the exact numbers and findings returned by your tools. Never hallucinate or estimate financial figures.
    4. Missing Data Protocol: If a tool returns "No transactions found", state clearly that the data isn't available. Do not invent or guess reasons why the data is missing. Simply offer a logical next step.
    5. Clean Professionalism: Keep your responses structured and scannable. Prioritize clear, easy-to-read financial breakdowns.
    6. Strictly do not use emojis. No one likes them.
    7. If the user uses a greeting, respond with a greeting. If the user uses a farewell, respond with a farewell. Always match the user's tone and style, but no emojis.
    8. Do not give user example bank account names. If the user does not specify a bank, ask them which bank they would like to analyze. Do not guess or assume."""),
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


def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("==================================================")
    print("BudAI Chat Initialized")
    print("==================================================")
    chat_history = []
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        output = response["output"]
        print(f"\nBudAI: {output}")
        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=output)
        ])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = sys.argv[1]
        response = agent_executor.invoke(
            {"input": user_input, "chat_history": []})
        print(response["output"])
    else:
        main()
