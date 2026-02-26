from Categorizer_Agent.CategorizerAgent import CategorizerAgent
from Forecaster_Agent.ForecasterAgent import ForecasterAgent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from dotenv import load_dotenv
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


load_dotenv()


@tool
def generate_financial_forecast(days: int = 60) -> str:
    """
    Runs the BudAI mathematical forecaster to predict future bank balances.

    Args:
        days (int): The number of days into the future to forecast. 
    """
    try:
        agent = ForecasterAgent()

        real_balance = agent.fetch_live_balance()
        S0, mu, sigma = agent.fetch_and_calculate_parameters(real_balance, 60)

        acc_id = agent.user_acc.account_id if agent.user_acc else "default_account"

        agent.run_cpp_simulation(acc_id, S0, mu, sigma, days=days, paths=1000)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(
            current_dir, "Forecaster_Agent", "all_paths_for_" + str(days) + "_days.csv")
        df = pd.read_csv(csv_path, header=None)

        return (
            f"Current Balance: £{S0:.2f}\n"
            f"Forecast for {days} days:\n"
            f"- Careless: £{df.iloc[0].iloc[-1]:.2f}\n"
            f"- Expected: £{df.iloc[1].iloc[-1]:.2f}\n"
            f"- Optimal: £{df.iloc[2].iloc[-1]:.2f}"
        )
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool
def classify_financial_data(from_date: str, to_date: str) -> str:
    """
    Runs the BudAI XGBoost Classifier to classify and categorize user's transaction between the given dates.

    Args:
        - from_date (str): Starting date for transactions.
        - to_date (str): End date for transactions.
    """
    try:
        agent = CategorizerAgent()
        df = agent.execute_cycle(from_date, to_date)

        if df is None or df.empty:
            return f"No transactions found between {from_date} and {to_date}."

        category_counts = df['Category'].value_counts().to_dict()
        summary = "\n".join(
            [f"- {cat}: {count} transactions" for cat, count in category_counts.items()])

        return f"Successfully classified {len(df)} transactions from {from_date} to {to_date}.\nCategory breakdown:\n{summary}"
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool
def find_total_spent_for_given_category(category_name: str) -> str:
    """
    Reads the categorized data after it has been categorized and finds the total spending in the requested category within the given timeline.

    Args:
        - category_name (str): Name of the category to find the total spendings in.
    """
    try:
        df = pd.read_csv("Categorizer_Agent/categorized_data.csv")
        category_df = df[df["Category"].str.lower() == category_name.lower()]
        total_spent = category_df["Amount"].abs().sum()
        number_of_transactions = len(category_df)

        return f"\nYou spend £{total_spent:.2f} in a total of {number_of_transactions} transactions for the {category_name} category.\n"

    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
    base_url="http://localhost:11434"
)

tools = [generate_financial_forecast, classify_financial_data,
         find_total_spent_for_given_category]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are BudAI, a friendly, empathetic, and highly capable personal financial coach. 

    YOUR CONVERSATIONAL RULES:
    1. Validate the User: Always start by warmly acknowledging the user's request. Make them feel heard and supported.
    2. Conversational Delivery: Do not just spit out raw data. Weave the numbers into natural, easy-to-read sentences. 
    3. The "Sandwich" Method: 
       - Start with a supportive opening.
       - Present the tool's data clearly (use light bullet points if there's a lot of data, but keep it conversational).
       - End with an encouraging closing statement or a gentle, actionable piece of advice.
    4. Tone: Keep it optimistic and professional. You are talking to a friend who asked for financial help. 
    
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
    print("==================================================")
    print("BudAI Chat Initialized (AgentExecutor Mode)")
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
