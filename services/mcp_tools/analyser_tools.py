import logging
import pandas as pd
from langchain_core.tools import tool
from services.mcp_tools.shared_utils import (
    FindTotalSpentInput, FindHighestSpendingCategoryInput,
    PlotExpensesInput, PlotCashFlowMixedInput,
    _cache_chart_data, _parse_accounts, _get_combined_categorized_data
)
from services.Analyser_Agent.expense_analysis import ExpenseAnalysis
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

@tool(args_schema=PlotExpensesInput)
def plot_expenses(user_uuid: str, plot_time_type: str, from_date: str, to_date: str, account_ids: list[str]) -> str:
    """Show user's daily/weekly/monthly past expenditure between the said dates."""
    try:
        accounts, _ = _parse_accounts(account_ids, user_uuid)
        payload = []
        data_summary = []
        for acc in accounts:
            ea = ExpenseAnalysis(identifier=acc, user_uuid=user_uuid)
            if ea.fetch_data(from_date, to_date):
                plot_type = plot_time_type.lower()
                if plot_type == 'daily':
                    df_temp = ea.get_daily_spend_data()
                elif plot_type == 'weekly':
                    df_temp = ea.get_weekly_spend_data()
                else:
                    df_temp = ea.get_monthly_spend_data()
                bank_data = []
                d_key = 'date' if 'date' in df_temp.columns else 'Date'
                a_key = 'amount' if 'amount' in df_temp.columns else 'Amount'
                
                total_acc_spend = df_temp[a_key].abs().sum()
                data_summary.append(f"- {acc}: Total £{total_acc_spend:.2f} across {len(df_temp)} {plot_time_type} intervals.")
                
                for _, row in df_temp.iterrows():
                    bank_data.append({
                        "Date": row[d_key].isoformat() if hasattr(row[d_key], 'isoformat') else str(row[d_key]),
                        "Amount": round(float(row[a_key]), 2)
                    })
                payload.append({"bank_name": acc, "data": bank_data})
        
        if not payload:
            return "No data found for the selected accounts and date range."
            
        cache_id = _cache_chart_data(payload)
        summary_text = "\n".join(data_summary)
        return f"Expense plot generated. [TRIGGER_HISTORICAL_{plot_time_type.upper()}_CHART:{cache_id}]\n\nDATA SUMMARY:\n{summary_text}"
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Error: {str(e)}"

@tool(args_schema=FindTotalSpentInput)
def find_total_spent_for_given_category(user_uuid: str, category: str, account_ids: list[str], from_date: str = None, to_date: str = None) -> str:
    """Calculate the total amount of money spent by the user within a specific given category."""
    try:
        accounts, suffix = _parse_accounts(account_ids, user_uuid)
        df = _get_combined_categorized_data(accounts, suffix, user_uuid, from_date, to_date)
        if df.empty:
            return "Error: No data."
        cat_key = 'category' if 'category' in df.columns else 'Category'
        amt_key = 'amount' if 'amount' in df.columns else 'Amount'
        if category.lower() == "all":
            category_totals = []
            for cat in df[cat_key].unique():
                cat_df = df[df[cat_key] == cat]
                category_totals.append(f"- {cat}: £{cat_df[amt_key].abs().sum():.2f}")
            return "Breakdown:\n" + "\n".join(category_totals)
        else:
            cat_df = df[df[cat_key].str.lower() == category.lower()]
            if cat_df.empty:
                return f"No transactions for {category}."
            total = cat_df[amt_key].abs().sum()
            return f"Total spent in {category}: £{total:.2f}"
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Error: {str(e)}"

@tool(args_schema=FindHighestSpendingCategoryInput)
def find_highest_spending_category(user_uuid: str, account_ids: list[str], from_date: str = None, to_date: str = None) -> str:
    """Identify the single spending category where the user has spent the maximum amount of money."""
    try:
        accounts, suffix = _parse_accounts(account_ids, user_uuid)
        df = _get_combined_categorized_data(accounts, suffix, user_uuid, from_date, to_date)
        if df.empty:
            return "Error: No data."
        cat_key = 'category' if 'category' in df.columns else 'Category'
        amt_key = 'amount' if 'amount' in df.columns else 'Amount'
        expenses_df = df[df[cat_key].str.lower() != 'income'].copy()
        if expenses_df.empty:
            return "No expense categories found."
        grouped = expenses_df.groupby(cat_key)[amt_key].apply(lambda x: x.abs().sum()).reset_index()
        highest = grouped.loc[grouped[amt_key].idxmax()]
        return f"Your highest spending category is {highest[cat_key]} with £{highest[amt_key]:.2f}."
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Error: {str(e)}"

@tool(args_schema=PlotCashFlowMixedInput)
def plot_cash_flow_mixed(user_uuid: str, account_ids: list[str], from_date: str, to_date: str) -> str:
    """Generate a cash flow visualization showing daily net income vs expenses."""
    try:
        accounts, _ = _parse_accounts(account_ids, user_uuid)
        from services.api_integrator.get_account_detail import UserAccounts
        payload = []
        data_summary = []
        for acc in accounts:
            user_acc = UserAccounts(user_id=user_uuid)
            df = user_acc.get_bank_transactions(acc, user_uuid, from_date, to_date)
            if df.empty:
                continue
            df['date'] = pd.to_datetime(df['date'], format='ISO8601', utc=True)
            df['Month'] = df['date'].dt.strftime('%b %Y')
            monthly = df.groupby('Month')['amount'].agg(
                Income=lambda x: x[x > 0].sum(),
                Expense=lambda x: x[x < 0].abs().sum(),
                Net_Balance='sum'
            ).reset_index()
            bank_data = []
            
            acc_total_income = monthly['Income'].sum()
            acc_total_expense = monthly['Expense'].sum()
            data_summary.append(f"- {acc}: Total Income £{acc_total_income:.2f}, Total Expenses £{acc_total_expense:.2f}, Net £{acc_total_income - acc_total_expense:.2f}")
            
            for _, row in monthly.iterrows():
                bank_data.append({
                    "Month": row['Month'],
                    "Income": round(float(row['Income']), 2),
                    "Expense": round(float(row['Expense']), 2),
                    "Net_Balance": round(float(row['Net_Balance']), 2)
                })
            payload.append({"bank_name": acc, "data": bank_data})
            
        if not payload:
            return "No cash flow data available for the requested period."
            
        cache_id = _cache_chart_data(payload)
        summary_text = "\n".join(data_summary)
        return f"Cash flow chart generated. [TRIGGER_CASH_FLOW_CHART:{cache_id}]\n\nDATA SUMMARY:\n{summary_text}"
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Error: {str(e)}"
