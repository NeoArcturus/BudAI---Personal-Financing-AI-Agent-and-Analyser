import pandas as pd
import numpy as np
import os
from Categorizer_Agent.CategorizerAgent import CategorizerAgent
from Forecaster_Agent.ForecasterAgent import ForecasterAgent
from Analyser_Agent.category_dist import CategoryDistribution
from Analyser_Agent.expense_analysis import ExpenseAnalysis
from langchain_core.tools import tool
from api_integrator.get_account_detail import UserAccount


@tool
def generate_financial_forecast(days: int = 60, bank_name_or_id: str = None) -> str:
    """
    Runs the BudAI mathematical forecaster to predict future bank balances using a hybrid Heston-Jump Diffusion model.

    Args:
        days (int): The number of days into the future to forecast. 
        bank_name_or_id (str, optional): The name of the bank (e.g., 'Monzo') or the specific account ID.
    """
    try:
        agent = ForecasterAgent()
        real_balance = agent.fetch_live_balance(bank_name_or_id)
        print(
            f"Account balance retrieved for forecasting: £{real_balance:.2f}")
        S0, mu, _ = agent.fetch_and_calculate_parameters(real_balance, 60)

        csv_path = agent.run_hybrid_simulation(
            agent.user_acc.account_id, S0, mu, days=days, paths=1000000)

        df = pd.read_csv(csv_path, header=None)
        final_balances = df.iloc[:, -1]

        careless_val = final_balances.quantile(0.05, interpolation='nearest')
        expected_val = final_balances.mean()
        optimal_val = final_balances.quantile(0.95, interpolation='nearest')

        agent.analyze_and_plot(csv_path, S0, threshold_pct=0.5)

        return (
            f"Current Balance: £{S0:.2f}\n"
            f"Forecast for {days} days based on 1000000 stochastic simulations:\n"
            f"- Careless (5th Percentile): £{careless_val:.2f}\n"
            f"- Expected (Mean): £{expected_val:.2f}\n"
            f"- Optimal (95th Percentile): £{optimal_val:.2f}\n"
        )
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool
def classify_financial_data(from_date: str, to_date: str, bank_name_or_id: str = None) -> str:
    """
    Runs the BudAI XGBoost Classifier to classify and categorize user's transaction between the given dates.

    Args:
        from_date (str): Starting date for transactions.
        to_date (str): End date for transactions.
        bank_name_or_id (str, optional): The name of the bank (e.g., 'Monzo') or the specific account ID.
    """
    try:
        agent = CategorizerAgent()
        df = agent.execute_cycle(bank_name_or_id, from_date, to_date)

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
        )
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool
def find_total_spent_for_given_category(category_name: str, bank_name_or_id: str) -> str:
    """
    Reads the categorized data and finds the total spending amount-wise. If the user asks for a specific category, pass that name.
    If the user asks for ALL categories or a full breakdown, pass "all" as the category_name.

    Args:
        category_name (str): Name of the category, or "all" to get a summary of all categories.
        bank_name_or_id (str): The name of the bank (e.g., 'Monzo', 'Wise') or the specific account ID. REQUIRED.
    """
    if not bank_name_or_id or bank_name_or_id.lower() == "none":
        return "CRITICAL: You must ask the user which bank account they want to check spending for."

    try:
        from api_integrator.get_account_detail import UserAccount
        user_acc = UserAccount(bank_name_or_id)

        if not user_acc.account_id:
            return f"Error: Could not resolve '{bank_name_or_id}' to a valid linked account."

        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(
            current_dir, "saved_media", "csvs", f"categorized_data_{user_acc.account_id}.csv")

        if not os.path.exists(csv_path):
            return "Error: No categorized data found. Please run the classify_financial_data tool first."

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
                current_dir, "saved_media", "csvs", f"total_per_category_{user_acc.account_id}.csv")
            totals_df.to_csv(totals_csv_path, index=False)

            summary_lines = []
            for item in category_totals:
                if item["Is_Income"]:
                    summary_lines.append(
                        f"- {item['Category']}: +£{item['Total_Amount']:.2f} ({item['Transaction_Count']} transactions)")
                else:
                    summary_lines.append(
                        f"- {item['Category']}: £{item['Total_Amount']:.2f} ({item['Transaction_Count']} transactions)")

            return "Total spending breakdown for all categories (Sorted from highest to lowest):\n" + "\n".join(summary_lines)

        else:
            category_df = df[df["Category"].str.lower() ==
                             category_name.lower()]
            if category_df.empty:
                return f"No transactions found for the '{category_name}' category."

            total_spent = category_df["Amount"].abs().sum()
            number_of_transactions = len(category_df)
            return f"\nYou spent £{total_spent:.2f} in a total of {number_of_transactions} transactions for the {category_name} category.\n"

    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool
def find_highest_spending_category(bank_name_or_id: str) -> str:
    """
    Reads the aggregated category totals CSV to find the single category where the user spent the maximum amount of money.

    Args:
        bank_name_or_id (str): The name of the bank (e.g., 'Monzo', 'Wise') or the specific account ID. REQUIRED.
    """
    if not bank_name_or_id or bank_name_or_id.lower() == "none":
        return "CRITICAL: You must ask the user which bank account they want to check."

    try:
        from api_integrator.get_account_detail import UserAccount
        user_acc = UserAccount(bank_name_or_id)

        if not user_acc.account_id:
            return f"Error: Could not resolve '{bank_name_or_id}' to a valid linked account."

        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(
            current_dir, "saved_media", "csvs", f"total_per_category_{user_acc.account_id}.csv")

        if not os.path.exists(csv_path):
            return "Error: No aggregated totals found. Please run the find_total_spent_for_given_category tool for 'all' categories first."

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
            f"- Number of Transactions: {highest_category['Transaction_Count']}"
        )

    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool
def create_bargraph_chart_and_save() -> str:
    """
    Create a bar graph chart of all categories and save the plot.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(
            current_dir, "saved_media", "csvs", "categorized_data.csv")
        df = pd.read_csv(csv_path)
        bar_chart_analyser = CategoryDistribution(df)
        bar_chart_analyser.save_bar_plot()
        return f"Your bar chart has been created."
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool
def plot_expenses(plot_time_type: str, from_date: str, to_date: str, bank_name_or_id: str = None) -> str:
    """
    Plots the user's expense between the given dates on a daily/weekly/monthly basis.

    Args:
        plot_time_type (str): Expense plot time type - Daily, Weekly or Monthly
        from_date (str): Starting date for transactions.
        to_date (str): End date for transactions.
        bank_name_or_id (str, optional): The name of the bank (e.g., 'Monzo') or the specific account ID.
    """
    try:
        expense_analyser = ExpenseAnalysis(identifier=bank_name_or_id)
        has_data = expense_analyser.fetch_data(from_date, to_date)

        if not has_data:
            return f"No expense transactions found to plot between {from_date} and {to_date}."

        plot_type_lower = plot_time_type.lower()
        if plot_type_lower == "daily":
            expense_analyser.plot_daily_spend()
            return f"Your daily expenditure between {from_date} and {to_date} has been plotted!\n"
        elif plot_type_lower == "weekly":
            expense_analyser.plot_weekly_spend()
            return f"Your weekly expenditure between {from_date} and {to_date} has been plotted!\n"
        elif plot_type_lower == "monthly":
            expense_analyser.plot_monthly_spend()
            return f"Your monthly expenditure between {from_date} and {to_date} has been plotted!\n"
        else:
            return "Invalid plot time type. Please specify Daily, Weekly, or Monthly."
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool
def generate_expense_forecast(days: int = 30, bank_name_or_id: str = None) -> str:
    """
    Runs the BudAI mathematical forecaster to predict future expenses.

    Args:
        days (int): The number of days into the future to forecast. 
        bank_name_or_id (str, optional): The name of the bank (e.g., 'Monzo') or the specific account ID.
    """
    try:
        agent = ForecasterAgent()
        agent.user_acc = UserAccount(bank_name_or_id)
        E0, mu_E = agent.fetch_expense_parameters(lookback_days=60)

        csv_path = agent.run_expense_simulation(
            agent.user_acc.account_id, E0, mu_E, days=days, paths=1000000)

        df = pd.read_csv(csv_path, header=None)
        convergent_path = df.iloc[0]
        converged_total = convergent_path.iloc[1:].sum()

        t_days = np.arange(1, days + 1)
        historical_expected_total = (E0 * np.exp(mu_E * t_days)).sum()
        actual_err = abs(converged_total -
                         historical_expected_total) / historical_expected_total

        agent.analyze_and_plot_expenses(csv_path, E0, mu_E, days=days)

        return (
            f"BudAI Quantitative Convergence Report:\n"
            f"Based on past data, your historical expected spend over {days} days is £{historical_expected_total:.2f}.\n\n"
            f"The C++ engine scanned 1000000 futures in-memory and locked onto the exact timeline that converges to your baseline.\n\n"
            f"- Converged Path Total: £{converged_total:.2f}\n"
            f"- Path Convergence Error: {(actual_err * 100):.4f}%\n"
            f"- Highest projected daily spend on this path: £{convergent_path.max():.2f}\n"
            f"- Lowest projected daily spend on this path: £{convergent_path.min():.2f}\n"
        )
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"
