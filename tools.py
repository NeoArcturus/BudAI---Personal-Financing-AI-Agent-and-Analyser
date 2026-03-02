import pandas as pd
import os
from dotenv import load_dotenv
from Categorizer_Agent.CategorizerAgent import CategorizerAgent
from Forecaster_Agent.ForecasterAgent import ForecasterAgent
from langchain_core.tools import tool


@tool
def generate_financial_forecast(days: int = 60) -> str:
    """
    Runs the BudAI mathematical forecaster to predict future bank balances using a hybrid Heston-Jump Diffusion model.

    Args:
        days (int): The number of days into the future to forecast. 
    """
    try:
        agent = ForecasterAgent()

        real_balance = agent.fetch_live_balance()
        S0, mu, _ = agent.fetch_and_calculate_parameters(real_balance, 60)

        acc_id = agent.user_acc.account_id if agent.user_acc else "default_account"

        agent.run_hybrid_simulation(
            acc_id, S0, mu, days=days, paths=50000)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(
            current_dir, "Forecaster_Agent", "mathematics", "hybrid_paths.csv")

        if not os.path.exists(csv_path):
            csv_path = os.path.join(current_dir, "hybrid_paths.csv")

        df = pd.read_csv(csv_path, header=None)

        final_balances = df.iloc[:, -1]

        careless_val = final_balances.quantile(0.05, interpolation='nearest')
        expected_val = final_balances.mean()
        optimal_val = final_balances.quantile(0.95, interpolation='nearest')

        agent.analyze_and_plot(csv_path, S0, threshold_pct=0.5)

        return (
            f"Current Balance: £{S0:.2f}\n"
            f"Forecast for {days} days based on 50,000 stochastic simulations:\n"
            f"- Careless (5th Percentile): £{careless_val:.2f}\n"
            f"- Expected (Mean): £{expected_val:.2f}\n"
            f"- Optimal (95th Percentile): £{optimal_val:.2f}"
            f"\n"
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

        return (
            f"Successfully classified {len(df)} transactions from {from_date} to {to_date}.\nCategory breakdown:\n{summary}"
            f"\n"
        )
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
