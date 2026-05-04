import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from langchain_core.tools import tool
from config import SessionLocal
from sqlalchemy import text

from services.mcp_tools.tool_schema import (
    GenerateFinancialForecastInput, ClassifyFinancialDataInput, FindTotalSpentInput,
    FindHighestSpendingCategoryInput, CreateBargraphChartInput, CreatePieChartInput,
    PlotExpensesInput, GenerateExpenseForecastInput, AnalyzeCriticalSurvivalMetricsInput,
    AnalyzeWealthAccelerationMetricsInput, PlotCashFlowMixedInput, PlotHealthRadarInput,
    UpdateTransactionCategoryInput, RetrainCategorizerInput, _cache_chart_data,
    _parse_accounts, _check_and_handle_sca_error, _get_combined_categorized_data
)

from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
from services.Forecaster_Agent.ForecasterAgent import ForecasterAgent
from services.Analyser_Agent.expense_analysis import ExpenseAnalysis
from services.api_integrator.get_account_detail import UserAccounts
from services.Analyser_Agent.financial_health import FinancialHealthAnalyzer

logger = logging.getLogger("uvicorn.error")


@tool(args_schema=GenerateFinancialForecastInput)
def generate_financial_forecast(bank_name_or_id: str, user_uuid: str, days: int = 60) -> str:
    """Predict the user's future bank account balances over a specified number of days."""
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
            suffix = ",".join(suffixes)

        if not accounts:
            return "CRITICAL TOOL ERROR: No valid accounts found."

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        for acc in accounts:
            try:
                UserAccounts(user_id=user_uuid).get_transactions(
                    acc, user_uuid, start_date, end_date)
            except Exception as e:
                sca_msg = _check_and_handle_sca_error(e, acc, user_uuid)
                if sca_msg:
                    return sca_msg

        agent = ForecasterAgent()
        combined_expected = []
        combined_careless = []
        combined_optimal = []
        total_real_balance = 0

        payload = []

        for acc in accounts:
            real_balance = agent.fetch_live_balance(acc, user_uuid)
            total_real_balance += real_balance
            S0, mu, _ = agent.fetch_and_calculate_parameters(
                acc, real_balance, user_uuid, 60)
            df_temp = agent.run_hybrid_simulation(
                acc, S0, mu, days=days, paths=1000)

            bank_data = []
            if not df_temp.empty:
                exp_vals = df_temp.iloc[0].values.tolist()
                care_vals = df_temp.iloc[1].values.tolist()
                opt_vals = df_temp.iloc[2].values.tolist()

                if not combined_expected:
                    combined_expected = exp_vals
                    combined_careless = care_vals
                    combined_optimal = opt_vals
                else:
                    combined_expected = [x + y for x,
                                         y in zip(combined_expected, exp_vals)]
                    combined_careless = [x + y for x,
                                         y in zip(combined_careless, care_vals)]
                    combined_optimal = [x + y for x,
                                        y in zip(combined_optimal, opt_vals)]

                for i in range(days):
                    bank_data.append({
                        "Day": f"Day {i}",
                        "Expected Balance": round(exp_vals[i], 2),
                        "Careless Scenario (5%)": round(care_vals[i], 2),
                        "Optimal Scenario (95%)": round(opt_vals[i], 2)
                    })
            payload.append({"bank_name": acc, "data": bank_data})

        cache_id = _cache_chart_data(payload)

        final_expected = combined_expected[-1] if combined_expected else 0
        final_careless = combined_careless[-1] if combined_careless else 0
        final_optimal = combined_optimal[-1] if combined_optimal else 0
        net_change = final_expected - total_real_balance
        trajectory_status = "POSITIVE" if net_change > 0 else "NEGATIVE"

        return (
            f"--- DATA SUMMARY FOR BUDAI ---\n"
            f"Current Balance: £{total_real_balance:.2f}\n"
            f"- Expected (Mean): £{final_expected:.2f}\n"
            f"- Careless (5th Percentile): £{final_careless:.2f}\n"
            f"- Optimal (95th Percentile): £{final_optimal:.2f}\n"
            f"Analytical Context: The user's expected {days}-day trajectory is {trajectory_status}, with a net change of £{net_change:.2f}. "
            f"Please summarize this trajectory and the risk between the careless and optimal scenarios.\n"
            f"[TRIGGER_BALANCE_CHART:{cache_id}:{days}]"
        )
    except Exception as e:
        if "⚠️ **SCA" in str(e):
            return str(e)
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=ClassifyFinancialDataInput)
def classify_financial_data(from_date: str, to_date: str, bank_name_or_id: str, user_uuid: str) -> str:
    """Categorize and classify the user's raw bank transactions into distinct spending categories."""
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
            suffix = ",".join(suffixes)

        agent = CategorizerAgent()
        combined_df = pd.DataFrame()

        for acc in accounts:
            try:
                df = agent.execute_cycle(acc, user_uuid, from_date, to_date)
                if df is not None and not df.empty:
                    df['bank_name'] = acc
                    combined_df = pd.concat(
                        [combined_df, df], ignore_index=True)
            except Exception as e:
                sca_msg = _check_and_handle_sca_error(e, acc, user_uuid)
                if sca_msg:
                    return sca_msg

        if combined_df.empty:
            return f"No transactions found between {from_date} and {to_date}."

        anomalies_count = len(
            combined_df[combined_df['Category'] == 'High-Risk / Anomaly'])
        anomaly_alert = ""
        if anomalies_count > 0:
            anomaly_alert = f"CRITICAL PRIORITY: {anomalies_count} 'High-Risk / Anomaly' transactions detected! Explicitly warn the user to review these immediately.\n"

        payload = []
        for acc in accounts:
            acc_df = combined_df[combined_df['bank_name'] ==
                                 acc] if 'bank_name' in combined_df.columns else combined_df
            bank_data = []
            for cat in acc_df['Category'].unique():
                cat_df = acc_df[acc_df['Category'] == cat]
                bank_data.append({
                    "Category": str(cat),
                    "Total_Amount": round(float(cat_df['Amount'].abs().sum()), 2),
                    "Transaction_Count": len(cat_df)
                })
            bank_data.sort(key=lambda x: x["Total_Amount"], reverse=True)
            payload.append({"bank_name": acc, "data": bank_data})

        cache_id = _cache_chart_data(payload)
        category_counts = combined_df['Category'].value_counts().to_dict()
        summary = "\n".join(
            [f"- {cat}: {count} transactions" for cat, count in category_counts.items()])
        top_category = list(category_counts.keys())[
            0] if category_counts else "None"
        report = agent.get_classification_report()

        return (
            f"--- DATA SUMMARY FOR BUDAI ---\n"
            f"{anomaly_alert}"
            f"Successfully classified {len(combined_df)} transactions from {from_date} to {to_date}.\n"
            f"Category breakdown:\n{summary}\n\n"
            f"--- AI Classifier Metrics ---\n{report}\n"
            f"Analytical Context: The most frequent transaction category is '{top_category}'. "
            f"Present the breakdown cleanly to the user and ask if they want to analyze their spending in '{top_category}' further.\n"
            f"[TRIGGER_CATEGORIZED_CHART:{cache_id}:{from_date}|{to_date}]"
        )
    except Exception as e:
        if "⚠️ **SCA" in str(e):
            return str(e)
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=FindTotalSpentInput)
def find_total_spent_for_given_category(category_name: str, bank_name_or_id: str, user_uuid: str) -> str:
    """Calculate the total amount of money spent by the user within a specific given category."""
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
            suffix = ",".join(suffixes)

        df = _get_combined_categorized_data(accounts, suffix, user_uuid)

        if df is None or df.empty:
            return "Error: No categorized data found."

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
            summary_lines = []
            for item in category_totals:
                sign = "+£" if item["Is_Income"] else "£"
                summary_lines.append(
                    f"- {item['Category']}: {sign}{item['Total_Amount']:.2f} ({item['Transaction_Count']} tx)")

            return (f"--- DATA SUMMARY FOR BUDAI ---\n"
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
        if "⚠️ **SCA" in str(e):
            return str(e)
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=FindHighestSpendingCategoryInput)
def find_highest_spending_category(bank_name_or_id: str, user_uuid: str) -> str:
    """Identify the single spending category where the user has spent the maximum amount of money."""
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
            suffix = ",".join(suffixes)

        df = _get_combined_categorized_data(accounts, suffix, user_uuid)

        if df is None or df.empty:
            return "Error: No categorized data found."

        expenses_df = df[df['Category'].str.lower() != 'income'].copy()
        if expenses_df.empty:
            return "No expense categories found in the data."

        grouped = expenses_df.groupby('Category')['Amount'].apply(
            lambda x: x.abs().sum()).reset_index()
        highest_category = grouped.loc[grouped['Amount'].idxmax()]
        total_expenses = grouped['Amount'].sum()
        percentage = (
            highest_category['Amount'] / total_expenses) * 100 if total_expenses > 0 else 0

        return (
            f"--- DATA SUMMARY FOR BUDAI ---\n"
            f"Highest Spending Category:\n"
            f"- Category: {highest_category['Category']}\n"
            f"- Total Spent: £{highest_category['Amount']:.2f}\n"
            f"- Percentage of Total Expense: {percentage:.1f}%\n"
            f"Analytical Context: Inform the user of their highest category and explicitly mention that it makes up {percentage:.1f}% of their total categorized spending. Ask if they want to reduce this."
        )
    except Exception as e:
        if "⚠️ **SCA" in str(e):
            return str(e)
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=CreateBargraphChartInput)
def create_bargraph_chart_and_save(bank_name_or_id: str, user_uuid: str) -> str:
    """Generate a visual distribution chart of the user's categorized spending."""
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
            suffix = ",".join(suffixes)

        df = _get_combined_categorized_data(accounts, suffix, user_uuid)

        if df is None or df.empty:
            return "No categorized data available to chart. The categorizer could not process the transactions."

        payload = []
        for acc in accounts:
            acc_df = df[df['bank_name'] ==
                        acc] if 'bank_name' in df.columns else df
            bank_data = []
            for cat in acc_df['Category'].unique():
                cat_df = acc_df[acc_df['Category'] == cat]
                bank_data.append({
                    "Category": str(cat),
                    "Total_Amount": round(float(cat_df['Amount'].abs().sum()), 2),
                    "Transaction_Count": len(cat_df)
                })
            bank_data.sort(key=lambda x: x["Total_Amount"], reverse=True)
            payload.append({"bank_name": acc, "data": bank_data})

        cache_id = _cache_chart_data(payload)

        expenses_df = df[df['Category'].str.lower() != 'income'].copy()
        grouped = expenses_df.groupby('Category')['Amount'].apply(
            lambda x: x.abs().sum()).reset_index()
        grouped = grouped.sort_values(by='Amount', ascending=False)
        top_categories = grouped.head(3).to_dict('records')
        top_str = ", ".join(
            [f"{c['Category']} (£{c['Amount']:.2f})" for c in top_categories])

        return (f"--- DATA SUMMARY FOR BUDAI ---\n"
                f"Top Expense Categories to mention: {top_str}\n"
                f"Analytical Context: Instruct the UI to render the chart and summarize the top 3 spending categories to the user.\n"
                f"[TRIGGER_CATEGORIZED_CHART:{cache_id}]\n")
    except Exception as e:
        if "⚠️ **SCA" in str(e):
            return str(e)
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=CreatePieChartInput)
def create_pie_chart_and_save(bank_name_or_id: str, user_uuid: str) -> str:
    """Generate a visual pie/doughnut chart representing the proportional distribution of the user's categorized spending."""
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
            suffix = ",".join(suffixes)

        df = _get_combined_categorized_data(accounts, suffix, user_uuid)

        if df is None or df.empty:
            return "No categorized data available to chart. The categorizer could not process the transactions."

        payload = []
        for acc in accounts:
            acc_df = df[df['bank_name'] ==
                        acc] if 'bank_name' in df.columns else df
            bank_data = []
            for cat in acc_df['Category'].unique():
                cat_df = acc_df[acc_df['Category'] == cat]
                bank_data.append({
                    "Category": str(cat),
                    "Total_Amount": round(float(cat_df['Amount'].abs().sum()), 2),
                    "Transaction_Count": len(cat_df)
                })
            bank_data.sort(key=lambda x: x["Total_Amount"], reverse=True)
            payload.append({"bank_name": acc, "data": bank_data})

        cache_id = _cache_chart_data(payload)

        expenses_df = df[df['Category'].str.lower() != 'income'].copy()
        grouped = expenses_df.groupby('Category')['Amount'].apply(
            lambda x: x.abs().sum()).reset_index()
        grouped = grouped.sort_values(by='Amount', ascending=False)
        top_categories = grouped.head(3).to_dict('records')
        top_str = ", ".join(
            [f"{c['Category']} (£{c['Amount']:.2f})" for c in top_categories])

        return (f"--- DATA SUMMARY FOR BUDAI ---\n"
                f"Top Expense Categories to mention: {top_str}\n"
                f"Analytical Context: Instruct the UI to render the pie chart and summarize the top 3 spending categories to the user.\n"
                f"[TRIGGER_CATEGORIZED_DOUGHNUT_CHART:{cache_id}]\n")
    except Exception as e:
        if "⚠️ **SCA" in str(e):
            return str(e)
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=PlotExpensesInput)
def plot_expenses(plot_time_type: str, from_date: str, to_date: str, bank_name_or_id: str, user_uuid: str) -> str:
    """Show user's daily/weekly/monthly past expenditure between the said dates."""
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
            suffix = ",".join(suffixes)

        if not accounts:
            return "CRITICAL TOOL ERROR: No valid accounts found."

        for acc in accounts:
            try:
                UserAccounts(user_id=user_uuid).get_transactions(
                    acc, user_uuid, from_date, to_date)
            except Exception as e:
                sca_msg = _check_and_handle_sca_error(e, acc, user_uuid)
                if sca_msg:
                    return sca_msg

        plot_type_lower = plot_time_type.lower()
        freq_map = {'daily': 'D', 'weekly': 'W-MON', 'monthly': 'ME'}

        if plot_type_lower not in freq_map:
            return "Invalid plot time type. Please specify Daily, Weekly, or Monthly."

        combined_df = pd.DataFrame()
        has_data = False
        payload = []

        for acc in accounts:
            ea = ExpenseAnalysis(identifier=acc, user_uuid=user_uuid)
            if ea.fetch_data(from_date, to_date):
                has_data = True
                if plot_type_lower == 'daily':
                    df_temp = ea.get_daily_spend_data()
                elif plot_type_lower == 'weekly':
                    df_temp = ea.get_weekly_spend_data()
                else:
                    df_temp = ea.get_monthly_spend_data()

                df_temp['bank_name'] = acc

                bank_data = []
                for _, row in df_temp.iterrows():
                    date_val = row['date'] if 'date' in df_temp.columns else row['Date']
                    amt_val = row['amount'] if 'amount' in df_temp.columns else row['Amount']

                    if pd.notnull(date_val):
                        bank_data.append({
                            "Date": pd.to_datetime(date_val).strftime('%Y-%m-%d'),
                            "Amount": round(float(amt_val), 2)
                        })
                payload.append({"bank_name": acc, "data": bank_data})

                if combined_df.empty:
                    combined_df = df_temp
                else:
                    combined_df = pd.concat(
                        [combined_df, df_temp], ignore_index=True)

        if not has_data:
            return f"No expense transactions found between {from_date} and {to_date}."

        cache_id = _cache_chart_data(payload)

        date_col = 'date' if 'date' in combined_df.columns else 'Date'
        amt_col = 'amount' if 'amount' in combined_df.columns else 'Amount'

        if len(accounts) > 1:
            grouped = combined_df.groupby([date_col, 'bank_name'])[
                amt_col].sum().reset_index()
            pivot_df = grouped.pivot(
                index=date_col, columns='bank_name', values=amt_col).fillna(0).reset_index()
            numeric_cols = [c for c in pivot_df.columns if c != date_col]
            total_spend = pivot_df[numeric_cols].sum().sum()
            period_totals = pivot_df[numeric_cols].sum(axis=1)
            avg_spend = period_totals.mean()
            highest_idx = period_totals.idxmax()
            highest_date = pivot_df.loc[highest_idx, date_col]
            highest_amt = period_totals.max()
        else:
            grouped = combined_df.groupby(
                date_col)[amt_col].sum().reset_index()
            total_spend = grouped[amt_col].sum()
            avg_spend = grouped[amt_col].mean()
            highest_row = grouped.loc[grouped[amt_col].idxmax()]
            highest_date = highest_row[date_col]
            highest_amt = highest_row[amt_col]

        return (f"--- DATA SUMMARY FOR BUDAI ---\n"
                f"Total Spend in Period: £{total_spend:.2f}\n"
                f"Average {plot_time_type.capitalize()} Spend: £{avg_spend:.2f}\n"
                f"Peak Spending Period: {highest_date} at £{highest_amt:.2f}\n"
                f"Analytical Context: Provide the user with these specific numbers. Point out the peak spending period.\n"
                f"[TRIGGER_HISTORICAL_{plot_time_type.upper()}_CHART:{cache_id}:{from_date}|{to_date}]\n")
    except Exception as e:
        if "⚠️ **SCA" in str(e):
            return str(e)
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=GenerateExpenseForecastInput)
def generate_expense_forecast(bank_name_or_id: str, user_uuid: str, days: int = 30) -> str:
    """Predict the user's future spending habits and expected expenses over a specified number of days."""
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
            suffix = ",".join(suffixes)

        if not accounts:
            return "CRITICAL TOOL ERROR: No valid accounts found."

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        for acc in accounts:
            try:
                UserAccounts(user_id=user_uuid).get_transactions(
                    acc, user_uuid, start_date, end_date)
            except Exception as e:
                sca_msg = _check_and_handle_sca_error(e, acc, user_uuid)
                if sca_msg:
                    return sca_msg

        agent = ForecasterAgent()
        combined_expected = []
        total_historical_expected = 0
        total_converged = 0

        payload = []

        for acc in accounts:
            E0, mu_E = agent.fetch_expense_parameters(
                acc, user_uuid, lookback_days=60)
            df_temp = agent.run_expense_simulation(
                acc, E0, mu_E, days=days, paths=1000)

            bank_data = []
            if not df_temp.empty:
                convergent_path = df_temp.iloc[0].values.tolist()

                if not combined_expected:
                    combined_expected = convergent_path
                else:
                    combined_expected = [
                        x + y for x, y in zip(combined_expected, convergent_path)]

                total_converged += sum(convergent_path[1:])
                t_days = np.arange(1, days + 1)
                total_historical_expected += (E0 * np.exp(mu_E * t_days)).sum()

                for i in range(days):
                    bank_data.append(
                        {"Day": f"Day {i}", "Projected Daily Spend (£)": round(convergent_path[i], 2)})

            payload.append({"bank_name": acc, "data": bank_data})

        cache_id = _cache_chart_data(payload)
        actual_err = abs(total_converged - total_historical_expected) / \
            total_historical_expected if total_historical_expected > 0 else 0
        trajectory_insight = "higher" if total_converged > total_historical_expected else "lower"

        return (
            f"--- DATA SUMMARY FOR BUDAI ---\n"
            f"Based on past data, expected spend over {days} days is £{total_historical_expected:.2f}.\n\n"
            f"- Converged Path Total: £{total_converged:.2f}\n"
            f"- Path Convergence Error: {(actual_err * 100):.4f}%\n"
            f"Analytical Context: The simulated convergence path suggests total spending might be {trajectory_insight} than the rigid historical baseline. Present the expected spend of £{total_historical_expected:.2f} and offer advice on how to keep costs down over the next {days} days.\n"
            f"[TRIGGER_EXPENSE_CHART:{cache_id}:{days}]"
        )
    except Exception as e:
        if "⚠️ **SCA" in str(e):
            return str(e)
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=AnalyzeCriticalSurvivalMetricsInput)
def analyze_critical_survival_metrics(user_uuid: str) -> str:
    """Analyze the user's critical financial safety and baseline survival metrics."""
    try:
        accounts, suffix = _parse_accounts("ALL", user_uuid)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        for acc in accounts:
            try:
                UserAccounts(user_id=user_uuid).get_transactions(
                    acc, user_uuid, start_date, end_date)
            except Exception as e:
                sca_msg = _check_and_handle_sca_error(e, acc, user_uuid)
                if sca_msg:
                    return sca_msg

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
        if "⚠️ **SCA" in str(e):
            return str(e)
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=AnalyzeWealthAccelerationMetricsInput)
def analyze_wealth_acceleration_metrics(user_uuid: str) -> str:
    """Analyze the user's wealth-building efficiency and financial growth metrics."""
    try:
        accounts, suffix = _parse_accounts("ALL", user_uuid)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        for acc in accounts:
            try:
                UserAccounts(user_id=user_uuid).get_transactions(
                    acc, user_uuid, start_date, end_date)
            except Exception as e:
                sca_msg = _check_and_handle_sca_error(e, acc, user_uuid)
                if sca_msg:
                    return sca_msg

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
        if "⚠️ **SCA" in str(e):
            return str(e)
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=PlotCashFlowMixedInput)
def plot_cash_flow_mixed(bank_name_or_id: str, user_uuid: str) -> str:
    """Generate a mixed chart visualizing the user's historical cash flow, comparing income against expenses."""
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
            suffix = ",".join(suffixes)

        start_date = (datetime.now() - timedelta(days=365)
                      ).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        for acc in accounts:
            try:
                UserAccounts(user_id=user_uuid).get_transactions(
                    acc, user_uuid, start_date, end_date)
            except Exception as e:
                sca_msg = _check_and_handle_sca_error(e, acc, user_uuid)
                if sca_msg:
                    return sca_msg

        with SessionLocal() as session:
            if str(bank_name_or_id).upper() == "ALL":
                query = text(
                    "SELECT date, amount, b.bank_name FROM transactions t JOIN banks b ON t.bank_uuid = b.bank_uuid WHERE t.user_uuid = :user_uuid")
                df = pd.read_sql_query(query, session.connection(), params={
                                       "user_uuid": user_uuid})
            else:
                seq = ','.join([':acc' + str(i) for i in range(len(accounts))])
                query = text(f"""
                    SELECT t.date, t.amount, b.bank_name FROM transactions t
                    JOIN banks b ON t.bank_uuid = b.bank_uuid
                    WHERE b.bank_name IN ({seq}) AND t.user_uuid = :user_uuid
                """)
                params = {"user_uuid": user_uuid}
                for i, acc in enumerate(accounts):
                    params[f"acc{i}"] = acc
                df = pd.read_sql_query(
                    query, session.connection(), params=params)

        if df.empty:
            return "No transactions found to generate cash flow."

        payload = []
        for acc in accounts:
            acc_df = df[df['bank_name'] ==
                        acc] if 'bank_name' in df.columns else df
            acc_df['Date'] = pd.to_datetime(
                acc_df['date'], errors='coerce', utc=True)
            acc_df['Month'] = acc_df['Date'].dt.to_period('M')

            bank_data = []

            for period in sorted(acc_df['Month'].dropna().unique()):
                month_str = period.strftime('%Y-%m')
                month_df = acc_df[acc_df['Month'] == period]
                inc = float(month_df[month_df['amount'] > 0]['amount'].sum())
                exp = abs(
                    float(month_df[month_df['amount'] < 0]['amount'].sum()))
                bank_data.append({
                    "Month": month_str,
                    "Income": round(inc, 2),
                    "Expense": round(exp, 2),
                    "Net_Balance": round(inc - exp, 2)
                })
            payload.append({"bank_name": acc, "data": bank_data})

        cache_id = _cache_chart_data(payload)

        df['Date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        df['Month'] = df['Date'].dt.to_period('M')

        income = df[df['amount'] > 0].groupby('Month')['amount'].sum()
        expenses = df[df['amount'] < 0].groupby('Month')['amount'].sum().abs()

        total_income = income.sum()
        total_expense = expenses.sum()
        net = total_income - total_expense

        return (f"--- DATA SUMMARY FOR BUDAI ---\n"
                f"Historical Cash Flow calculated.\n"
                f"Total Income: £{total_income:.2f}\n"
                f"Total Expense: £{total_expense:.2f}\n"
                f"Net Balance: £{net:.2f}\n"
                f"[TRIGGER_CASH_FLOW_CHART:{cache_id}]")
    except Exception as e:
        if "⚠️ **SCA" in str(e):
            return str(e)
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=PlotHealthRadarInput)
def plot_health_radar(user_uuid: str) -> str:
    """Generate a visual radar chart profiling the user's overall financial health metrics."""
    try:
        accounts, suffix = _parse_accounts("ALL", user_uuid)
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        for acc in accounts:
            try:
                UserAccounts(user_id=user_uuid).get_transactions(
                    acc, user_uuid, start_date, end_date)
            except Exception as e:
                sca_msg = _check_and_handle_sca_error(e, acc, user_uuid)
                if sca_msg:
                    return sca_msg

        analyzer = FinancialHealthAnalyzer(user_uuid)

        runway = min(analyzer.calculate_liquid_runway() / 180 * 100, 100)
        nw_vel = max(
            min((analyzer.calculate_net_worth_velocity() + 1000) / 2000 * 100, 100), 0)
        mpc_score = max((1.0 - analyzer.calculate_mpc()) * 100, 0)
        absorption = min(analyzer.calculate_shock_absorption() / 5 * 100, 100)
        drag_score = max(100 - (analyzer.calculate_interest_drag() * 5), 0)

        payload = [{
            "bank_name": "Overall Portfolio",
            "data": [
                {"Metric": "Liquid Runway", "Score": round(runway, 2)},
                {"Metric": "Wealth Velocity", "Score": round(nw_vel, 2)},
                {"Metric": "Savings Efficiency", "Score": round(mpc_score, 2)},
                {"Metric": "Shock Absorption", "Score": round(absorption, 2)},
                {"Metric": "Debt Control", "Score": round(drag_score, 2)}
            ]
        }]
        cache_id = _cache_chart_data(payload)

        return (f"--- DATA SUMMARY FOR BUDAI ---\n"
                f"Health Scores Computed:\n"
                f"Liquid Runway: {runway:.2f}/100\n"
                f"Wealth Velocity: {nw_vel:.2f}/100\n"
                f"Savings Efficiency: {mpc_score:.2f}/100\n"
                f"Shock Absorption: {absorption:.2f}/100\n"
                f"Debt Control: {drag_score:.2f}/100\n"
                f"[TRIGGER_HEALTH_RADAR_CHART:{cache_id}]")
    except Exception as e:
        if "⚠️ **SCA" in str(e):
            return str(e)
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=UpdateTransactionCategoryInput)
def update_transaction_category(user_uuid: str, transaction_uuid: str, corrected_category: str) -> str:
    """Updates the category of a specific transaction based on user feedback and saves it to the database."""
    try:
        agent = CategorizerAgent()
        agent.save_manual_label(
            user_uuid, transaction_uuid, corrected_category)
        return f"Successfully updated transaction {transaction_uuid} to '{corrected_category}'. The database has been secured."
    except ValueError as ve:
        return f"Validation Error: {str(ve)}. Ensure the category matches the valid system rules."
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"


@tool(args_schema=RetrainCategorizerInput)
def retrain_categorization_model(user_uuid: str) -> str:
    """Retrains the underlying categorization model using the user's recent manual corrections."""
    try:
        agent = CategorizerAgent()
        result = agent.retrain_from_feedback(user_uuid)
        if result.get("trained"):
            return f"Successfully retrained model with {result.get('samples')} samples."
        return f"Retraining skipped: {result.get('reason')}"
    except Exception as e:
        return f"CRITICAL TOOL ERROR: {str(e)}"
