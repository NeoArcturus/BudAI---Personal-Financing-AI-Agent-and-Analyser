from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama
from tools import generate_financial_forecast, classify_financial_data, find_total_spent_for_given_category, create_bargraph_chart_and_save, generate_expense_forecast, plot_expenses
import os
from datetime import datetime

current_date = datetime.now().strftime("%B %d, %Y")

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    base_url="http://localhost:11434"
)

tools = [generate_financial_forecast, classify_financial_data,
         find_total_spent_for_given_category, create_bargraph_chart_and_save, generate_expense_forecast, plot_expenses]

prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are BudAI, a friendly, empathetic, and highly capable personal financing tool (Do not mention you are a tool or introduce yourself, just reply back naturally.).
     Current date: {current_date} 

    YOUR CONVERSATIONAL RULES:
    1. Validate the User: Always start by warmly acknowledging the user's request. Make them feel heard and supported.
    2. Conversational Delivery: Do not just spit out raw data. Weave the numbers into natural, easy-to-read sentences. Do not use emojis, no one likes them.
    3. The "Sandwich" Method: 
       - Start with a supportive opening.
       - Present the tool's data clearly (use light bullet points if there's a lot of data, but keep it conversational).
       - End with an encouraging closing statement or a gentle, actionable piece of advice.
    4. Tone: Keep it optimistic and professional. You are talking to a friend who asked for financial help.
    5. If a Critical Tool error occurs, do not disclose that to the user. Just mention "There's an internal issue. Thank you for being patient.
    6. While explaining, do not hallucinate. Use the output returned by the tools and use the exact wordings returned."
    
    CRITICAL: When the user asks for a forecast or to categorize data, you MUST use your tools first. Do not guess. Do not hallucinate."""),
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
    main()
