import pandas as pd
import numpy as np
import os
import sqlite3
from datetime import datetime, timedelta
from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
from services.Forecaster_Agent.ForecasterAgent import ForecasterAgent
from services.Analyser_Agent.category_dist import CategoryDistribution
from services.Analyser_Agent.expense_analysis import ExpenseAnalysis
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from services.api_integrator.get_account_detail import UserAccounts
from services.Analyser_Agent.financial_health import FinancialHealthAnalyzer


class GenerateFinancialForecastInput(BaseModel):
    bank_name_or_id: str = Field(...,
                                 description="MUST be 'ALL' unless explicitly named.")
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")
    days: int = Field(
        60, description="The number of days into the future to forecast.")


class ClassifyFinancialDataInput(BaseModel):
    from_date: str = Field(...,
                           description="Starting date for transactions in YYYY-MM-DD format.")
    to_date: str = Field(...,
                         description="End date for transactions in YYYY-MM-DD format.")
    bank_name_or_id: str = Field(...,
                                 description="MUST be 'ALL' unless explicitly named.")
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


class FindTotalSpentInput(BaseModel):
    category_name: str = Field(...,
                               description="Name of the category, or 'all'.")
    bank_name_or_id: str = Field(...,
                                 description="MUST be 'ALL' unless explicitly named.")
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


class FindHighestSpendingCategoryInput(BaseModel):
    bank_name_or_id: str = Field(...,
                                 description="MUST be 'ALL' unless explicitly named.")
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


class CreateBargraphChartInput(BaseModel):
    bank_name_or_id: str = Field(...,
                                 description="MUST be 'ALL' unless explicitly named.")
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


class PlotExpensesInput(BaseModel):
    plot_time_type: str = Field(...,
                                description="MUST be exactly 'Daily', 'Weekly', or 'Monthly'.")
    from_date: str = Field(...,
                           description="Starting date for transactions in YYYY-MM-DD format.")
    to_date: str = Field(...,
                         description="End date for transactions in YYYY-MM-DD format.")
    bank_name_or_id: str = Field(...,
                                 description="MUST be 'ALL' unless explicitly named.")
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


class GenerateExpenseForecastInput(BaseModel):
    bank_name_or_id: str = Field(...,
                                 description="MUST be 'ALL' unless explicitly named.")
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")
    days: int = Field(
        30, description="The number of days into the future to forecast.")


class AnalyzeCriticalSurvivalMetricsInput(BaseModel):
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


class AnalyzeWealthAccelerationMetricsInput(BaseModel):
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


class PlotCashFlowMixedInput(BaseModel):
    bank_name_or_id: str = Field(...,
                                 description="MUST be 'ALL' unless explicitly named.")
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


class PlotHealthRadarInput(BaseModel):
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


def _parse_accounts(bank_name_or_id, user_uuid):
    if not bank_name_or_id or str(bank_name_or_id).lower() in ["none", ""]:
        return [], ""

    if str(bank_name_or_id).upper() == "ALL":
        with sqlite3.connect("budai_memory.db") as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT b.bank_name FROM banks b WHERE b.user_uuid = ?", (user_uuid,))
            accounts = [row[0] for row in cursor.fetchall()]
        return accounts, "ALL"
    else:
        with sqlite3.connect("budai_memory.db") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT a.account_id, b.bank_name 
                FROM accounts a 
                JOIN banks b ON a.bank_uuid = b.bank_uuid 
                WHERE (b.bank_name = ? OR a.account_id = ?) AND a.user_uuid = ?
            """, (bank_name_or_id, bank_name_or_id, user_uuid))
            row = cursor.fetchone()

            if row:
                return [row[1]], row[0]

            return [bank_name_or_id], bank_name_or_id


def _get_combined_categorized_data(accounts, suffix, user_uuid):
    print(
        f"[BACKEND LOG] _get_combined_categorized_data | accounts: {accounts} | suffix: {suffix} | user_uuid: {user_uuid}")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(current_dir, '..', "saved_media", "csvs")
    os.makedirs(csv_dir, exist_ok=True)

    combined_df = pd.DataFrame()
    agent = CategorizerAgent()

    start_date = "2010-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    for acc in accounts:
        try:
            df = agent.execute_cycle(acc, user_uuid, start_date, end_date)
            if df is not None and not df.empty:
                df['bank_name'] = acc
                combined_df = pd.concat([combined_df, df], ignore_index=True)

            if len(accounts) > 1:
                ind_csv = os.path.join(csv_dir, f"categorized_data_{acc}.csv")
                if os.path.exists(ind_csv):
                    try:
                        os.remove(ind_csv)
                    except Exception:
                        pass
        except Exception as e:
            print(
                f"[BACKEND LOG] Error in _get_combined_categorized_data for {acc}: {e}")
            pass

    if combined_df.empty:
        return None

    combined_csv_path = os.path.join(csv_dir, f"categorized_data_{suffix}.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    return combined_csv_path


@tool(args_schema=GenerateFinancialForecastInput)
def generate_financial_forecast(bank_name_or_id: str, user_uuid: str, days: int = 60) -> str:
    """
    Predict the user's future bank account balances over a specified number of days. Use this tool to show projected balance trajectories and assess expected financial growth or decline over time.
    """
    print(
        f"[BACKEND LOG] generate_financial_forecast | bank_name_or_id: {bank_name_or_id} | days: {days} | user_uuid: {user_uuid}")
    try:
        if str(bank_name_or_id).upper() == "ALL":
            accounts, suffix = _parse_accounts("ALL", user_uuid)
        else:
            raw_banks = [b.strip()
                         for b in str(bank_name_or_id).split(",") if b.strip()]
            accounts = []
            suffixes = []
            for rb in raw_banks:
                accs, suff = _parse_accounts(rb, user_uuid)
                accounts.extend(accs)
                suffixes.append(suff)
            suffix = "COMBINED" if len(suffixes) > 1 else suffixes[0]

        if not accounts:
            return "CRITICAL TOOL ERROR: No valid accounts found."

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        for acc in accounts:
            try:
                UserAccounts(user_id=user_uuid).get_transactions(
                    acc, user_uuid, start_date, end_date)
            except Exception:
                pass

        agent = ForecasterAgent()
        combined_rows = []
        total_real_balance = 0

        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_dir = os.path.join(current_dir, "..", "saved_media", "csvs")
        os.makedirs(csv_dir, exist_ok=True)

        for acc in accounts:
            real_balance = agent.fetch_live_balance(acc, user_uuid)
            total_real_balance += real_balance
            S0, mu, _ = agent.fetch_and_calculate_parameters(
                acc, real_balance, user_uuid, 60)

            temp_csv = agent.run_hybrid_simulation(
                acc, S0, mu, days=days, paths=100000)
            df_temp = pd.read_csv(temp_csv, header=None)

            if not df_temp.empty:
                if len(combined_rows) == 0:
                    combined_rows = df_temp.values.tolist()
                else:
                    for i in range(len(combined_rows)):
                        for j in range(len(combined_rows[i])):
                            combined_rows[i][j] += df_temp.values[i][j]

            if len(accounts) > 1:
                try:
                    os.remove(temp_csv)
                except Exception:
                    pass

        final_csv_path = os.path.join(csv_dir, f"hybrid_paths_{suffix}.csv")
        pd.DataFrame(combined_rows).to_csv(
            final_csv_path, index=False, header=False)

        df_final = pd.read_csv(final_csv_path, header=None)
        final_expected = df_final.iloc[0, -1]
        final_careless = df_final.iloc[1, -1]
        final_optimal = df_final.iloc[2, -1]

        net_change = final_expected - total_real_balance
        trajectory_status = "POSITIVE" if net_change > 0 else "NEGATIVE"

        return (
            f"--- DATA SUMMARY FOR BUDAI ---\n"
            f"Forecast data available at: {final_csv_path}\n"
            f"Current Balance: £{total_real_balance:.2f}\n"
            f"- Expected (Mean): £{final_expected:.2f}\n"
            f"- Careless (5th Percentile): £{final_careless:.2f}\n"
            f"- Optimal (95th Percentile): £{final_optimal:.2f}\n"
            f"Analytical Context: The user's expected {days}-day trajectory is {trajectory_status}, with a net change of £{net_change:.2f}. "
            f"Please summarize this trajectory and the risk between the careless and optimal scenarios.\n"
            f"[TRIGGER_BALANCE_CHART]"
        )
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=ClassifyFinancialDataInput)
def classify_financial_data(from_date: str, to_date: str, bank_name_or_id: str, user_uuid: str) -> str:
    """
    Categorize and classify the user's raw bank transactions into distinct spending categories. Use this tool when the user asks for a breakdown of their spending or wants to organize their recent financial activity.
    """
    print(
        f"[BACKEND LOG] classify_financial_data | from_date: {from_date} | to_date: {to_date} | bank_name_or_id: {bank_name_or_id} | user_uuid: {user_uuid}")
    try:
        if str(bank_name_or_id).upper() == "ALL":
            accounts, suffix = _parse_accounts("ALL", user_uuid)
        else:
            raw_banks = [b.strip()
                         for b in str(bank_name_or_id).split(",") if b.strip()]
            accounts = []
            suffixes = []
            for rb in raw_banks:
                accs, suff = _parse_accounts(rb, user_uuid)
                accounts.extend(accs)
                suffixes.append(suff)
            suffix = "COMBINED" if len(suffixes) > 1 else suffixes[0]

        agent = CategorizerAgent()
        combined_df = pd.DataFrame()

        for acc in accounts:
            try:
                df = agent.execute_cycle(acc, user_uuid, from_date, to_date)
                if df is not None and not df.empty:
                    combined_df = pd.concat(
                        [combined_df, df], ignore_index=True)
            except Exception as e:
                print(
                    f"[BACKEND LOG] Categorizer execution failed for {acc}: {e}")

        if combined_df.empty:
            return f"No transactions found between {from_date} and {to_date}."

        try:
            bar_chart_analyser = CategoryDistribution(combined_df)
            bar_chart_analyser.extract_category_distribution_data(suffix)
        except Exception as chart_err:
            print(
                f"[BACKEND LOG] Failed to generate category dist chart: {chart_err}")

        category_counts = combined_df['Category'].value_counts().to_dict()
        summary = "\n".join(
            [f"- {cat}: {count} transactions" for cat, count in category_counts.items()])

        top_category = list(category_counts.keys())[
            0] if category_counts else "None"
        report = agent.get_classification_report()

        return (
            f"--- DATA SUMMARY FOR BUDAI ---\n"
            f"Successfully classified {len(combined_df)} transactions from {from_date} to {to_date}.\n"
            f"Category breakdown:\n{summary}\n\n"
            f"--- AI Classifier Metrics ---\n{report}\n"
            f"Analytical Context: The most frequent transaction category is '{top_category}'. "
            f"Present the breakdown cleanly to the user and ask if they want to analyze their spending in '{top_category}' further.\n"
            f"[TRIGGER_CATEGORIZED_CHART]"
        )
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=FindTotalSpentInput)
def find_total_spent_for_given_category(category_name: str, bank_name_or_id: str, user_uuid: str) -> str:
    """
    Calculate the total amount of money spent by the user within a specific given category. Use this tool to answer direct questions about how much was spent on things like groceries, rent, or entertainment.
    """
    print(
        f"[BACKEND LOG] find_total_spent_for_given_category | category_name: {category_name} | bank_name_or_id: {bank_name_or_id} | user_uuid: {user_uuid}")
    if not bank_name_or_id or str(bank_name_or_id).lower() == "none":
        return "CRITICAL: Ask the user which bank account they want to check."

    try:
        if str(bank_name_or_id).upper() == "ALL":
            accounts, suffix = _parse_accounts("ALL", user_uuid)
        else:
            raw_banks = [b.strip()
                         for b in str(bank_name_or_id).split(",") if b.strip()]
            accounts = []
            suffixes = []
            for rb in raw_banks:
                accs, suff = _parse_accounts(rb, user_uuid)
                accounts.extend(accs)
                suffixes.append(suff)
            suffix = "COMBINED" if len(suffixes) > 1 else suffixes[0]

        csv_path = _get_combined_categorized_data(accounts, suffix, user_uuid)

        if not csv_path or not os.path.exists(csv_path):
            return "Error: No categorized data found."

        df = pd.read_csv(csv_path)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_dir = os.path.join(current_dir, '..', "saved_media", "csvs")

        if category_name.lower() == "all":
            category_totals = []
            total_outflow = 0
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
                if cat.lower() != "income":
                    total_outflow += total_spent

            category_totals.sort(key=lambda x: x["Total_Amount"], reverse=True)
            totals_df = pd.DataFrame(category_totals)
            totals_csv_path = os.path.join(
                csv_dir, f"total_per_category_{suffix}.csv")
            totals_df.to_csv(totals_csv_path, index=False)

            summary_lines = []
            for item in category_totals:
                sign = "+£" if item["Is_Income"] else "£"
                summary_lines.append(
                    f"- {item['Category']}: {sign}{item['Total_Amount']:.2f} ({item['Transaction_Count']} tx)")

            return (f"--- DATA SUMMARY FOR BUDAI ---\n"
                    f"Total spending breakdown CSV available at: {totals_csv_path}\n"
                    f"Total Categorized Outflow: £{total_outflow:.2f}\n"
                    f"Breakdown:\n" + "\n".join(summary_lines) +
                    f"\nAnalytical Context: Summarize these totals for the user, focusing on the top 2 highest expense categories.\n")
        else:
            category_df = df[df["Category"].str.lower() ==
                             category_name.lower()]
            if category_df.empty:
                return f"No transactions found for the '{category_name}' category."

            total_spent = category_df["Amount"].abs().sum()
            number_of_transactions = len(category_df)
            avg_tx = total_spent / number_of_transactions if number_of_transactions > 0 else 0

            return (f"--- DATA SUMMARY FOR BUDAI ---\n"
                    f"Category: {category_name}\n"
                    f"Total Spent: £{total_spent:.2f}\n"
                    f"Transaction Count: {number_of_transactions}\n"
                    f"Average Transaction Size: £{avg_tx:.2f}\n"
                    f"Analytical Context: Provide these statistics to the user conversationally. "
                    f"If the average transaction size is high, suggest they review those specific purchases.")
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=FindHighestSpendingCategoryInput)
def find_highest_spending_category(bank_name_or_id: str, user_uuid: str) -> str:
    """
    Identify the single spending category where the user has spent the maximum amount of money. Use this tool to quickly highlight the user's biggest financial drain or top expense area.
    """
    print(
        f"[BACKEND LOG] find_highest_spending_category | bank_name_or_id: {bank_name_or_id} | user_uuid: {user_uuid}")
    try:
        if str(bank_name_or_id).upper() == "ALL":
            accounts, suffix = _parse_accounts("ALL", user_uuid)
        else:
            raw_banks = [b.strip()
                         for b in str(bank_name_or_id).split(",") if b.strip()]
            accounts = []
            suffixes = []
            for rb in raw_banks:
                accs, suff = _parse_accounts(rb, user_uuid)
                accounts.extend(accs)
                suffixes.append(suff)
            suffix = "COMBINED" if len(suffixes) > 1 else suffixes[0]

        _get_combined_categorized_data(accounts, suffix, user_uuid)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(
            current_dir, "..", "saved_media", "csvs", f"total_per_category_{suffix}.csv")

        if not os.path.exists(csv_path):
            return "Error: No aggregated totals found."

        df = pd.read_csv(csv_path)
        expenses_df = df[df['Is_Income'] == False]

        if expenses_df.empty:
            return "No expense categories found in the data."

        highest_category = expenses_df.loc[expenses_df['Total_Amount'].idxmax(
        )]
        total_expenses = expenses_df['Total_Amount'].sum()
        percentage = (highest_category['Total_Amount'] /
                      total_expenses) * 100 if total_expenses > 0 else 0

        return (
            f"--- DATA SUMMARY FOR BUDAI ---\n"
            f"Highest Spending Category:\n"
            f"- Category: {highest_category['Category']}\n"
            f"- Total Spent: £{highest_category['Total_Amount']:.2f}\n"
            f"- Percentage of Total Expense: {percentage:.1f}%\n"
            f"Analytical Context: Inform the user of their highest category and explicitly mention that it makes up {percentage:.1f}% of their total categorized spending. Ask if they want to reduce this."
        )
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=CreateBargraphChartInput)
def create_bargraph_chart_and_save(bank_name_or_id: str, user_uuid: str) -> str:
    """
    Generate a visual distribution chart of the user's categorized spending. Use this tool when the user explicitly asks to view a chart, graph, or visual breakdown of their top expense categories.
    """
    print(
        f"[BACKEND LOG] create_bargraph_chart_and_save | bank_name_or_id: {bank_name_or_id} | user_uuid: {user_uuid}")
    try:
        if str(bank_name_or_id).upper() == "ALL":
            accounts, suffix = _parse_accounts("ALL", user_uuid)
        else:
            raw_banks = [b.strip()
                         for b in str(bank_name_or_id).split(",") if b.strip()]
            accounts = []
            suffixes = []
            for rb in raw_banks:
                accs, suff = _parse_accounts(rb, user_uuid)
                accounts.extend(accs)
                suffixes.append(suff)
            suffix = "COMBINED" if len(suffixes) > 1 else suffixes[0]

        combined_csv_path = _get_combined_categorized_data(
            accounts, suffix, user_uuid)

        if not combined_csv_path or not os.path.exists(combined_csv_path):
            return "No categorized data available to chart. The categorizer could not process the transactions."

        df = pd.read_csv(combined_csv_path)
        bar_chart_analyser = CategoryDistribution(df)
        dist_csv = bar_chart_analyser.extract_category_distribution_data(
            suffix)

        dist_df = pd.read_csv(dist_csv)
        expenses_only = dist_df[dist_df['Category'].str.lower() != 'income']
        top_categories = expenses_only.head(3).to_dict('records')
        top_str = ", ".join(
            [f"{c['Category']} (£{c['Total_Amount']})" for c in top_categories])

        return (f"--- DATA SUMMARY FOR BUDAI ---\n"
                f"Category distribution data successfully generated at {dist_csv}.\n"
                f"Top Expense Categories to mention: {top_str}\n"
                f"Analytical Context: Instruct the UI to render the chart and summarize the top 3 spending categories to the user.\n"
                f"[TRIGGER_CATEGORIZED_CHART]\n")
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=PlotExpensesInput)
def plot_expenses(plot_time_type: str, from_date: str, to_date: str, bank_name_or_id: str, user_uuid: str) -> str:
    """
    Show user's daily/weekly/monthly past expenditure between the said dates. Use this tool to visualize historical spending trends and identify peak spending periods over a given timeframe.
    CRITICAL: NEVER use this tool for future dates or forecasts. For forecasts, use generate_expense_forecast instead.
    """
    print(
        f"[BACKEND LOG] plot_expenses | INIT | type: {plot_time_type} | from: {from_date} | to: {to_date} | id: {bank_name_or_id}")
    try:
        if str(bank_name_or_id).upper() == "ALL":
            accounts, suffix = _parse_accounts("ALL", user_uuid)
        else:
            raw_banks = [b.strip()
                         for b in str(bank_name_or_id).split(",") if b.strip()]
            accounts = []
            suffixes = []
            for rb in raw_banks:
                accs, suff = _parse_accounts(rb, user_uuid)
                accounts.extend(accs)
                suffixes.append(suff)
            suffix = "COMBINED" if len(suffixes) > 1 else suffixes[0]

        print(
            f"[BACKEND LOG] plot_expenses | Parsed accounts: {accounts} | Suffix: {suffix}")

        if not accounts:
            return "CRITICAL TOOL ERROR: No valid accounts found."

        for acc in accounts:
            try:
                print(
                    f"[BACKEND LOG] plot_expenses | Syncing data for account: {acc}")
                UserAccounts(user_id=user_uuid).get_transactions(
                    acc, user_uuid, from_date, to_date)
            except Exception as e:
                print(
                    f"[BACKEND LOG] plot_expenses | Sync failed for {acc}: {e}")

        plot_type_lower = plot_time_type.lower()
        freq_map = {'daily': 'D', 'weekly': 'W-MON', 'monthly': 'ME'}
        prefix_map = {'daily': 'daily_spend',
                      'weekly': 'weekly_spend', 'monthly': 'monthly_spend'}

        if plot_type_lower not in freq_map:
            return "Invalid plot time type. Please specify Daily, Weekly, or Monthly."

        csv_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", "saved_media", "csvs")
        os.makedirs(csv_dir, exist_ok=True)

        combined_df = pd.DataFrame()
        has_data = False

        for acc in accounts:
            print(
                f"[BACKEND LOG] plot_expenses | Running ExpenseAnalysis for: {acc}")
            ea = ExpenseAnalysis(identifier=acc, user_uuid=user_uuid)
            if ea.fetch_data(from_date, to_date):
                has_data = True
                print(
                    f"[BACKEND LOG] plot_expenses | Data found for {acc}, processing exports")
                if plot_type_lower == 'daily':
                    temp_csv = ea.export_daily_spend_data()
                elif plot_type_lower == 'weekly':
                    temp_csv = ea.export_weekly_spend_data()
                else:
                    temp_csv = ea.export_monthly_spend_data()

                df_temp = pd.read_csv(temp_csv)
                df_temp['bank_name'] = acc

                if combined_df.empty:
                    combined_df = df_temp
                else:
                    combined_df = pd.concat(
                        [combined_df, df_temp], ignore_index=True)

                try:
                    os.remove(temp_csv)
                except Exception:
                    pass
            else:
                print(
                    f"[BACKEND LOG] plot_expenses | No data returned from ExpenseAnalysis for {acc}")

        if not has_data:
            print(
                "[BACKEND LOG] plot_expenses | ABORT: No expense transactions found across all accounts.")
            return f"No expense transactions found between {from_date} and {to_date}."

        print(
            f"[BACKEND LOG] plot_expenses | Combined DF constructed with shape: {combined_df.shape}")

        date_col = 'date' if 'date' in combined_df.columns else 'Date'
        amt_col = 'amount' if 'amount' in combined_df.columns else 'Amount'

        if len(accounts) > 1:
            grouped = combined_df.groupby([date_col, 'bank_name'])[
                amt_col].sum().reset_index()
            pivot_df = grouped.pivot(
                index=date_col, columns='bank_name', values=amt_col).fillna(0).reset_index()
            final_csv_path = os.path.join(
                csv_dir, f"{prefix_map[plot_type_lower]}_{suffix}.csv")
            pivot_df.to_csv(final_csv_path, index=False)

            numeric_cols = [c for c in pivot_df.columns if c != date_col]
            total_spend = pivot_df[numeric_cols].sum().sum()
            period_totals = pivot_df[numeric_cols].sum(axis=1)
            avg_spend = period_totals.mean()
            highest_idx = period_totals.idxmax()
            highest_date = pivot_df.loc[highest_idx, date_col]
            highest_amt = period_totals.max()
        else:
            final_csv_path = os.path.join(
                csv_dir, f"{prefix_map[plot_type_lower]}_{suffix}.csv")
            grouped = combined_df.groupby(
                date_col)[amt_col].sum().reset_index()
            grouped.to_csv(final_csv_path, index=False)
            total_spend = grouped[amt_col].sum()
            avg_spend = grouped[amt_col].mean()
            highest_row = grouped.loc[grouped[amt_col].idxmax()]
            highest_date = highest_row[date_col]
            highest_amt = highest_row[amt_col]

        print(
            f"[BACKEND LOG] plot_expenses | SUCCESS | Saved to {final_csv_path} | Total Spend: {total_spend}")

        return (f"--- DATA SUMMARY FOR BUDAI ---\n"
                f"Expense data for {plot_time_type} successfully generated at {final_csv_path}.\n"
                f"Total Spend in Period: £{total_spend:.2f}\n"
                f"Average {plot_time_type.capitalize()} Spend: £{avg_spend:.2f}\n"
                f"Peak Spending Period: {highest_date} at £{highest_amt:.2f}\n"
                f"Analytical Context: Provide the user with these specific numbers. Point out the peak spending period.\n"
                f"[TRIGGER_HISTORICAL_CHART]\n")
    except Exception as e:
        print(f"[BACKEND LOG] plot_expenses | FATAL EXCEPTION: {str(e)}")
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=GenerateExpenseForecastInput)
def generate_expense_forecast(bank_name_or_id: str, user_uuid: str, days: int = 30) -> str:
    """
    Predict the user's future spending habits and expected expenses over a specified number of days. Use this tool to forecast upcoming costs and visualize future spending trajectories.
    CRITICAL: You MUST use this tool if the user asks to predict, forecast, or CHART FUTURE expenses.
    Do NOT use plot_expenses for future forecasts.
    """
    print(
        f"[BACKEND LOG] generate_expense_forecast | days: {days} | bank_name_or_id: {bank_name_or_id} | user_uuid: {user_uuid}")
    try:
        if str(bank_name_or_id).upper() == "ALL":
            accounts, suffix = _parse_accounts("ALL", user_uuid)
        else:
            raw_banks = [b.strip()
                         for b in str(bank_name_or_id).split(",") if b.strip()]
            accounts = []
            suffixes = []
            for rb in raw_banks:
                accs, suff = _parse_accounts(rb, user_uuid)
                accounts.extend(accs)
                suffixes.append(suff)
            suffix = "COMBINED" if len(suffixes) > 1 else suffixes[0]

        if not accounts:
            return "CRITICAL TOOL ERROR: No valid accounts found."

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        for acc in accounts:
            try:
                UserAccounts(user_id=user_uuid).get_transactions(
                    acc, user_uuid, start_date, end_date)
            except Exception:
                pass

        agent = ForecasterAgent()
        combined_rows = []
        total_historical_expected = 0
        total_converged = 0

        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_dir = os.path.join(current_dir, "..", "saved_media", "csvs")
        os.makedirs(csv_dir, exist_ok=True)

        for acc in accounts:
            E0, mu_E = agent.fetch_expense_parameters(
                acc, user_uuid, lookback_days=60)
            temp_csv = agent.run_expense_simulation(
                acc, E0, mu_E, days=days, paths=100000)
            df_temp = pd.read_csv(temp_csv, header=None)
            convergent_path = df_temp.iloc[0].values.tolist()

            row_data = [acc] + \
                convergent_path if len(accounts) > 1 else convergent_path
            combined_rows.append(row_data)

            total_converged += sum(convergent_path[1:])
            t_days = np.arange(1, days + 1)
            total_historical_expected += (E0 * np.exp(mu_E * t_days)).sum()

            if len(accounts) > 1:
                try:
                    os.remove(temp_csv)
                except Exception:
                    pass

        final_csv_path = os.path.join(
            csv_dir, f"converged_expense_{suffix}.csv")
        pd.DataFrame(combined_rows).to_csv(
            final_csv_path, index=False, header=False)

        actual_err = abs(total_converged - total_historical_expected) / \
            total_historical_expected if total_historical_expected > 0 else 0
        trajectory_insight = "higher" if total_converged > total_historical_expected else "lower"

        return (
            f"--- DATA SUMMARY FOR BUDAI ---\n"
            f"BudAI Quantitative Convergence Report Data available at: {final_csv_path}\n"
            f"Based on past data, expected spend over {days} days is £{total_historical_expected:.2f}.\n\n"
            f"- Converged Path Total: £{total_converged:.2f}\n"
            f"- Path Convergence Error: {(actual_err * 100):.4f}%\n"
            f"Analytical Context: The simulated convergence path suggests total spending might be {trajectory_insight} than the rigid historical baseline. Present the expected spend of £{total_historical_expected:.2f} and offer advice on how to keep costs down over the next {days} days.\n"
            f"[TRIGGER_EXPENSE_CHART]"
        )
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=AnalyzeCriticalSurvivalMetricsInput)
def analyze_critical_survival_metrics(user_uuid: str) -> str:
    """
    Analyze the user's critical financial safety and baseline survival metrics. Use this tool to calculate emergency liquid runway, basic subsistence floor, and optimal debt repayment plans.
    """
    print(
        f"[BACKEND LOG] analyze_critical_survival_metrics | user_uuid: {user_uuid}")
    try:
        accounts, suffix = _parse_accounts("ALL", user_uuid)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        for acc in accounts:
            try:
                UserAccounts(user_id=user_uuid).get_transactions(
                    acc, user_uuid, start_date, end_date)
            except Exception:
                pass

        analyzer = FinancialHealthAnalyzer(user_uuid)
        runway = analyzer.calculate_liquid_runway()
        floor = analyzer.calculate_subsistence_floor()
        plan = analyzer.avalanche_debt_optimization(monthly_surplus=150.0)

        plan_str = "\n".join([f"- Route £{p['payment_allocation']:.2f} to {p['target']} (Saves £{p['interest_saved_annually']:.2f}/yr interest)" for p in plan]
                             ) if plan else "No active debt liabilities found."

        return (
            f"--- DATA SUMMARY FOR BUDAI ---\n"
            f"SURVIVAL METRICS\n"
            f"Liquid Runway: {runway:.1f} days\n"
            f"Subsistence Floor: £{floor:.2f}/month\n"
            f"Avalanche Debt Plan:\n{plan_str}\n"
            f"Analytical Context: Warn the user immediately if their runway is critically low (under 30 days). Present the subsistence floor and explain the math behind the debt avalanche routing.\n"
        )
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=AnalyzeWealthAccelerationMetricsInput)
def analyze_wealth_acceleration_metrics(user_uuid: str) -> str:
    """
    Analyze the user's wealth-building efficiency and financial growth metrics. Use this tool to calculate net worth velocity, marginal propensity to consume, and shock absorption capacity.
    """
    print(
        f"[BACKEND LOG] analyze_wealth_acceleration_metrics | user_uuid: {user_uuid}")
    try:
        accounts, suffix = _parse_accounts("ALL", user_uuid)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        for acc in accounts:
            try:
                UserAccounts(user_id=user_uuid).get_transactions(
                    acc, user_uuid, start_date, end_date)
            except Exception:
                pass

        analyzer = FinancialHealthAnalyzer(user_uuid)
        nw_velocity = analyzer.calculate_net_worth_velocity()
        mpc = analyzer.calculate_mpc()
        absorption = analyzer.calculate_shock_absorption()
        drag = analyzer.calculate_interest_drag()

        status = "GROWING" if nw_velocity > 0 else "DECLINING"

        return (
            f"--- DATA SUMMARY FOR BUDAI ---\n"
            f"WEALTH METRICS\n"
            f"Financial Status: {status}\n"
            f"Net Worth Velocity: £{nw_velocity:.2f}/month\n"
            f"Marginal Propensity to Consume (MPC): {mpc:.2f} ({(mpc*100):.0f}% of new income spent)\n"
            f"Shock Absorption Capacity: {absorption:.2f}x (Current Liquidity / Max Historical Monthly Deficit)\n"
            f"Interest Drag Ratio: {drag:.1f}%\n"
            f"Analytical Context: If Net Worth Velocity is negative, warn the user they are getting poorer. If MPC is near 1.0, warn about lifestyle creep. Use the Shock Absorption to rate their emergency fund resilience.\n"
        )
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=PlotCashFlowMixedInput)
def plot_cash_flow_mixed(bank_name_or_id: str, user_uuid: str) -> str:
    """
    Generate a mixed chart visualizing the user's historical cash flow, comparing income against expenses. Use this tool to show net balance trends and monthly financial inflow versus outflow.
    """
    print(
        f"[BACKEND LOG] plot_cash_flow_mixed | bank_name_or_id: {bank_name_or_id} | user_uuid: {user_uuid}")
    try:
        if str(bank_name_or_id).upper() == "ALL":
            accounts, suffix = _parse_accounts("ALL", user_uuid)
        else:
            raw_banks = [b.strip()
                         for b in str(bank_name_or_id).split(",") if b.strip()]
            accounts = []
            suffixes = []
            for rb in raw_banks:
                accs, suff = _parse_accounts(rb, user_uuid)
                accounts.extend(accs)
                suffixes.append(suff)
            suffix = "COMBINED" if len(suffixes) > 1 else suffixes[0]

        start_date = (datetime.now() - timedelta(days=365)
                      ).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        for acc in accounts:
            try:
                UserAccounts(user_id=user_uuid).get_transactions(
                    acc, user_uuid, start_date, end_date)
            except Exception:
                pass

        with sqlite3.connect("budai_memory.db") as conn:
            if suffix == "ALL":
                query = "SELECT date, amount FROM transactions WHERE user_uuid = ?"
                df = pd.read_sql_query(query, conn, params=(user_uuid,))
            else:
                query = """
                    SELECT t.date, t.amount FROM transactions t 
                    JOIN banks b ON t.bank_uuid = b.bank_uuid 
                    WHERE b.bank_name IN ({seq}) AND t.user_uuid = ?
                """.format(seq=','.join(['?']*len(accounts)))
                df = pd.read_sql_query(
                    query, conn, params=(*accounts, user_uuid))

        if df.empty:
            return "No transactions found to generate cash flow."

        df['Date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        df['Month'] = df['Date'].dt.to_period('M')

        income = df[df['amount'] > 0].groupby('Month')['amount'].sum()
        expenses = df[df['amount'] < 0].groupby('Month')['amount'].sum().abs()

        flow_df = pd.DataFrame(
            {'Income': income, 'Expense': expenses}).fillna(0).reset_index()
        flow_df['Month'] = flow_df['Month'].dt.strftime('%Y-%m')
        flow_df['Net_Balance'] = flow_df['Income'] - flow_df['Expense']

        csv_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", "saved_media", "csvs")
        csv_path = os.path.join(csv_dir, f"cash_flow_mixed_{suffix}.csv")
        flow_df.to_csv(csv_path, index=False)

        return (f"--- DATA SUMMARY FOR BUDAI ---\n"
                f"Cash flow data generated at {csv_path}.\n"
                f"[TRIGGER_CASH_FLOW_CHART]")
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=PlotHealthRadarInput)
def plot_health_radar(user_uuid: str) -> str:
    """
    Generate a visual radar chart profiling the user's overall financial health metrics. Use this tool to visually map out normalized scores for liquid runway, wealth velocity, savings efficiency, and debt control.
    """
    print(f"[BACKEND LOG] plot_health_radar | user_uuid: {user_uuid}")
    try:
        accounts, suffix = _parse_accounts("ALL", user_uuid)
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        for acc in accounts:
            try:
                UserAccounts(user_id=user_uuid).get_transactions(
                    acc, user_uuid, start_date, end_date)
            except Exception:
                pass

        analyzer = FinancialHealthAnalyzer(user_uuid)

        runway = min(analyzer.calculate_liquid_runway() / 180 * 100, 100)
        nw_vel = max(
            min((analyzer.calculate_net_worth_velocity() + 1000) / 2000 * 100, 100), 0)
        mpc_score = max((1.0 - analyzer.calculate_mpc()) * 100, 0)
        absorption = min(analyzer.calculate_shock_absorption() / 5 * 100, 100)
        drag_score = max(100 - (analyzer.calculate_interest_drag() * 5), 0)

        radar_data = [
            {"Metric": "Liquid Runway", "Score": runway},
            {"Metric": "Wealth Velocity", "Score": nw_vel},
            {"Metric": "Savings Efficiency", "Score": mpc_score},
            {"Metric": "Shock Absorption", "Score": absorption},
            {"Metric": "Debt Control", "Score": drag_score}
        ]

        csv_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", "saved_media", "csvs")
        csv_path = os.path.join(csv_dir, f"health_radar_{suffix}.csv")
        pd.DataFrame(radar_data).to_csv(csv_path, index=False)

        return (f"--- DATA SUMMARY FOR BUDAI ---\n"
                f"Radar chart generated at {csv_path}.\n"
                f"[TRIGGER_HEALTH_RADAR_CHART]")
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"
