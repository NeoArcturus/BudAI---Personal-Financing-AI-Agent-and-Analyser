# backend/services/agent_tools.py
import pandas as pd
import numpy as np
import os
from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
from services.Forecaster_Agent.ForecasterAgent import ForecasterAgent
from services.Analyser_Agent.category_dist import CategoryDistribution
from services.Analyser_Agent.expense_analysis import ExpenseAnalysis
from langchain_core.tools import tool
from services.api_integrator.get_account_detail import UserAccount


@tool
def generate_financial_forecast(bank_name_or_id: str, user_uuid: str = None, days: int = 60) -> str:
    """
    Runs the BudAI mathematical forecaster to predict future bank balances. Returns CSV path for frontend rendering.

    Args:
        days (int): The number of days into the future to forecast.
        bank_name_or_id (str): The name of the bank or specific account ID.
        user_uuid (str, optional): The user UUID.
    """
    try:
        agent = ForecasterAgent()
        agent.user_acc = UserAccount(bank_name_or_id, user_uuid)
        real_balance = agent.fetch_live_balance(
            agent.user_acc.account_id, user_uuid)
        S0, mu, _ = agent.fetch_and_calculate_parameters(
            real_balance, user_uuid, 60)

        csv_path = agent.run_hybrid_simulation(
            agent.user_acc.account_id, S0, mu, days=days, paths=1000000)

        df = pd.read_csv(csv_path, header=None)
        final_balances = df.iloc[:, -1]

        return (
            f"Forecast data available at: {csv_path}\n"
            f"Current Balance: £{real_balance:.2f}\n"
            f"- Expected (Mean): £{final_balances[0]:.2f}\n"
            f"- Careless (5th Percentile): £{final_balances[1]:.2f}\n"
            f"- Optimal (95th Percentile): £{final_balances[2]:.2f}\n"
            f"[TRIGGER_BALANCE_CHART]"
        )
    except ValueError as e:
        if str(e) == "MULTIPLE_ACCOUNTS":
            return "CRITICAL: The user has multiple linked bank accounts. Ask the user which account they want to run the forecast on."
        return f"CRITICAL TOOL ERROR: {str(e)}"
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool
def classify_financial_data(from_date: str, to_date: str, bank_name_or_id: str, user_uuid: str = None) -> str:
    """
    Runs the BudAI XGBoost Classifier.

    Args:
        from_date (str): Starting date for transactions.
        to_date (str): End date for transactions.
        bank_name_or_id (str, optional): The name of the bank or the specific account ID.
        user_uuid (str, optional): The user UUID.
    """
    try:
        agent = CategorizerAgent()
        df = agent.execute_cycle(
            bank_name_or_id, user_uuid, from_date, to_date)

        if df is None or df.empty:
            return f"No transactions found between {from_date} and {to_date}."

        category_counts = df['Category'].value_counts().to_dict()
        summary = "\n".join(
            [f"- {cat}: {count} transactions" for cat, count in category_counts.items()])

        report = agent.get_classification_report()

        return (
            f"Successfully classified {len(df)} transactions from {from_date} to {to_date}.\n"
            f"Category breakdown:\n{summary}\n\n"
            f"--- AI Classifier Metrics ---\n{report}\n"
            f"\n[TRIGGER_CATEGORIZED_CHART]"
        )

    except ValueError as e:
        if str(e) == "MULTIPLE_ACCOUNTS":
            return "CRITICAL: The user has multiple linked bank accounts. Ask the user which account they want to categorize data for."
        return f"CRITICAL TOOL ERROR: {str(e)}"
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool
def find_total_spent_for_given_category(category_name: str, bank_name_or_id: str, user_uuid: str = None) -> str:
    """
    Reads the categorized data and finds the total spending amount-wise.

    Args:
        category_name (str): Name of the category, or "all".
        bank_name_or_id (str): The name of the bank or specific account ID. REQUIRED.
        user_uuid (str, optional): The user UUID.
    """
    if not bank_name_or_id or bank_name_or_id.lower() == "none":
        return "CRITICAL: Ask the user which bank account they want to check."

    try:
        user_acc = UserAccount(bank_name_or_id, user_uuid)

        if not user_acc.account_id:
            return f"Error: Could not resolve '{bank_name_or_id}'."

        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, '..'))

        csv_path = os.path.join(
            root_dir, "saved_media", "csvs", f"categorized_data_{user_acc.account_id}.csv")

        if not os.path.exists(csv_path):
            return "Error: No categorized data found."

        df = pd.read_csv(csv_path)

        if category_name.lower() == "all":
            category_totals = []
            for cat in df['Category'].unique():
                cat_df = df[df['Category'] == cat]
                total_spent = cat_df["Amount"].abs().sum()
                count = len(cat_df)
                category_totals.append({
                    "Category": cat,
                    "Total_Amount": total_spent,
                    "Transaction_Count": count,
                    "Is_Income": cat.lower() == "income"
                })

            category_totals.sort(key=lambda x: x["Total_Amount"], reverse=True)
            totals_df = pd.DataFrame(category_totals)
            totals_csv_path = os.path.join(
                root_dir, "saved_media", "csvs", f"total_per_category_{user_acc.account_id}.csv")
            totals_df.to_csv(totals_csv_path, index=False)

            summary_lines = []
            for item in category_totals:
                sign = "+£" if item["Is_Income"] else "£"
                summary_lines.append(
                    f"- {item['Category']}: {sign}{item['Total_Amount']:.2f} ({item['Transaction_Count']} tx)")

            return f"Total spending breakdown CSV available at: {totals_csv_path}\n" + "\n".join(summary_lines)

        else:
            category_df = df[df["Category"].str.lower() ==
                             category_name.lower()]
            if category_df.empty:
                return f"No transactions found for the '{category_name}' category."

            total_spent = category_df["Amount"].abs().sum()
            number_of_transactions = len(category_df)
            return f"\nYou spent £{total_spent:.2f} in a total of {number_of_transactions} transactions for {category_name}.\n"

    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool
def find_highest_spending_category(bank_name_or_id: str, user_uuid: str = None) -> str:
    """
    Reads the aggregated category totals CSV to find the single category where the user spent the maximum amount of money.

    Args:
        bank_name_or_id (str): The name of the bank or specific account ID. REQUIRED.
        user_uuid (str, optional): The user UUID.
    """
    try:
        user_acc = UserAccount(bank_name_or_id, user_uuid)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(
            current_dir, "..", "saved_media", "csvs", f"total_per_category_{user_acc.account_id}.csv")

        if not os.path.exists(csv_path):
            return "Error: No aggregated totals found."

        df = pd.read_csv(csv_path)
        expenses_df = df[df['Is_Income'] == False]

        if expenses_df.empty:
            return "No expense categories found in the data."

        highest_category = expenses_df.loc[expenses_df['Total_Amount'].idxmax(
        )]

        return (
            f"Highest Spending Category:\n"
            f"- Category: {highest_category['Category']}\n"
            f"- Total Spent: £{highest_category['Total_Amount']:.2f}\n"
        )
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool
def create_bargraph_chart_and_save(bank_name_or_id: str, user_uuid: str = None) -> str:
    """
    Calculates category distribution and saves it to a CSV for frontend chart rendering.

    Args:
        bank_name_or_id (str): The name of the bank or specific account ID. REQUIRED.
        user_uuid (str, optional): The user UUID.
    """
    try:
        user_acc = UserAccount(bank_name_or_id, user_uuid)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, '..'))
        csv_path = os.path.join(
            root_dir, "saved_media", "csvs", "categorized_data.csv")
        df = pd.read_csv(csv_path)

        bar_chart_analyser = CategoryDistribution(df)
        dist_csv = bar_chart_analyser.extract_category_distribution_data(
            user_acc.account_id)
        return (f"Category distribution data successfully generated at {dist_csv}. Instruct the UI to render the chart.\n"
                f"\n[TRIGGER_CATEGORIZED_CHART]\n")
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool
def plot_expenses(plot_time_type: str, from_date: str, to_date: str, bank_name_or_id: str, user_uuid: str = None) -> str:
    """
    Generates CSV data of the user's expenses on a daily/weekly/monthly basis for frontend rendering.
    CRITICAL: NEVER use this tool for future dates or forecasts. For forecasts, use generate_expense_forecast instead.


    Args:
        plot_time_type(str): Expense plot time type - Daily, Weekly or Monthly
        from_date(str): Starting date for transactions.
        to_date(str): End date for transactions.
        bank_name_or_id(str): The name of the bank or specific account ID.
        user_uuid(str, optional): The user UUID.
    """
    try:
        expense_analyser = ExpenseAnalysis(
            identifier=bank_name_or_id, user_uuid=user_uuid)
        has_data = expense_analyser.fetch_data(from_date, to_date)

        if not has_data:
            return f"No expense transactions found between {from_date} and {to_date}."

        plot_type_lower = plot_time_type.lower()
        if plot_type_lower == "daily":
            csv_path = expense_analyser.export_daily_spend_data()
        elif plot_type_lower == "weekly":
            csv_path = expense_analyser.export_weekly_spend_data()
        elif plot_type_lower == "monthly":
            csv_path = expense_analyser.export_monthly_spend_data()
        else:
            return "Invalid plot time type. Please specify Daily, Weekly, or Monthly."

        return f"Expense data for {plot_time_type} successfully generated at {csv_path}. Instruct UI to render the chart.\n[TRIGGER_HISTORICAL_CHART]\n"

    except ValueError as e:
        if str(e) == "MULTIPLE_ACCOUNTS":
            return "CRITICAL: The user has multiple linked bank accounts. Ask the user which account they want to plot."
        return f"CRITICAL TOOL ERROR: {str(e)}"
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool
def generate_expense_forecast(days: int = 30, bank_name_or_id: str = None, user_uuid: str = None) -> str:
    """
    Runs the BudAI mathematical forecaster to predict future expenses. Returns CSV path for frontend rendering.
    CRITICAL: You MUST use this tool if the user asks to predict, forecast, or CHART FUTURE expenses.
    Do NOT use plot_expenses for future forecasts.

    Args:
        days(int): The number of days into the future to forecast.
        bank_name_or_id(str, optional): The name of the bank or specific account ID.
        user_uuid(str, optional): The user UUID.
    """
    try:
        agent = ForecasterAgent()
        agent.user_acc = UserAccount(bank_name_or_id, user_uuid)
        E0, mu_E = agent.fetch_expense_parameters(user_uuid, lookback_days=60)

        csv_path = agent.run_expense_simulation(
            agent.user_acc.account_id, E0, mu_E, days=days, paths=1000000)

        df = pd.read_csv(csv_path, header=None)
        convergent_path = df.iloc[0]
        converged_total = convergent_path.iloc[1:].sum()

        t_days = np.arange(1, days + 1)
        historical_expected_total = (E0 * np.exp(mu_E * t_days)).sum()
        actual_err = abs(converged_total -
                         historical_expected_total) / historical_expected_total

        return (
            f"BudAI Quantitative Convergence Report Data available at: {csv_path}\n"
            f"Based on past data, expected spend over {days} days is £{historical_expected_total:.2f}.\n\n"
            f"- Converged Path Total: £{converged_total:.2f}\n"
            f"- Path Convergence Error: {(actual_err * 100):.4f}%\n"
            f"\n[TRIGGER_EXPENSE_CHART]"
        )
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"
