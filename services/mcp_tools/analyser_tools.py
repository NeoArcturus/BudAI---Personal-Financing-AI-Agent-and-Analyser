import logging
import pandas as pd
from langchain_core.tools import tool
from services.mcp_tools.shared_utils import (
    FindTotalSpentInput, FindHighestSpendingCategoryInput,
    PlotExpensesInput, PlotCashFlowMixedInput,
    GetBudgetVarianceInput,
    _cache_chart_data, _parse_accounts, _get_combined_categorized_data
)
from services.Analyser_Agent.expense_analysis import ExpenseAnalysis
from services.Analyser_Agent.budget_engine import BudgetEngine
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)


@tool(args_schema=GetBudgetVarianceInput)
def get_budget_variance(user_uuid: str, category: str = None) -> str:
    """
    Calculate and return the spend velocity, projected spend, and variance for the user's budgets.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        category (str, optional): The specific budget category to analyze.
        
    Returns:
        str: A breakdown of budget variance, spending velocity, and projected end-of-month spend.
    """
    logger.info(f"Executing MCP Tool: get_budget_variance")
    try:
        engine = BudgetEngine(user_uuid)
        results = engine.get_variance_for_category(category)
        
        if not results:
            _res = "No active budgets found for this category."
            logger.info(f"Tool returned: {str(_res)[:1000]}")
            return _res
            
        lines = []
        for r in results:
            lines.append(f"Category: {r['category']}")
            lines.append(f"  Monthly Limit: £{r['monthly_limit']:.2f}")
            lines.append(f"  Spent So Far: £{r['spent_so_far']:.2f}")
            lines.append(f"  Spend Velocity: {r['velocity']}x")
            lines.append(f"  Projected Month-End Spend: £{r['projected_spend']:.2f}")
            lines.append(f"  Variance: £{r['variance']:.2f}")
            lines.append("")
        
        _res = "\n".join(lines)
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res
    except Exception as e:
        logger.error(f"Error calculating budget variance: {e}")
        _res = f"Error: {str(e)}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res

@tool(args_schema=PlotExpensesInput)
def plot_expenses(user_uuid: str, plot_time_type: str, from_date: str, to_date: str, account_ids: list[str]) -> str:
    """
    Show user's daily/weekly/monthly past expenditure between the said dates.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        plot_time_type (str): The interval type (e.g., daily, weekly, monthly).
        from_date (str): The start date in YYYY-MM-DD format.
        to_date (str): The end date in YYYY-MM-DD format.
        account_ids (list[str]): List of account IDs to analyze.
        
    Returns:
        str: A summary text and a chart trigger for the generated expense plot.
    """
    logger.info(f"Executing MCP Tool: plot_expenses")
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
                    data_point = {
                        "Date": row[d_key].isoformat() if hasattr(row[d_key], 'isoformat') else str(row[d_key]),
                        "Amount": round(float(row[a_key]), 2)
                    }
                    if 'description' in row:
                        data_point["descriptions"] = row['description']
                    if 'currency' in row and pd.notna(row['currency']):
                        data_point["currency"] = row['currency']
                    bank_data.append(data_point)
                payload.append({"bank_name": acc, "data": bank_data})
        
        if not payload:
            _res = "No data found for the selected accounts and date range."
            logger.info(f"Tool returned: {str(_res)[:1000]}")
            return _res
            
        cache_id = _cache_chart_data(payload)
        summary_text = "\n".join(data_summary)
        _res = f"Expense plot generated. [TRIGGER_HISTORICAL_{plot_time_type.upper()}_CHART:{cache_id}]\n\nDATA SUMMARY:\n{summary_text}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res
    except Exception as e:
        logger.error(f"Error: {e}")
        _res = f"Error: {str(e)}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res

@tool(args_schema=FindTotalSpentInput)
def find_total_spent_for_given_category(user_uuid: str, category: str, account_ids: list[str], from_date: str = None, to_date: str = None) -> str:
    """
    Calculate the total amount of money spent by the user within a specific given category.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        category (str): The transaction category to sum, or 'all'.
        account_ids (list[str]): List of account IDs to analyze.
        from_date (str, optional): The start date in YYYY-MM-DD format.
        to_date (str, optional): The end date in YYYY-MM-DD format.
        
    Returns:
        str: The total amount spent in the requested category.
    """
    logger.info(f"Executing MCP Tool: find_total_spent_for_given_category")
    try:
        accounts, suffix = _parse_accounts(account_ids, user_uuid)
        df = _get_combined_categorized_data(accounts, suffix, user_uuid, from_date, to_date)
        if df.empty:
            _res = "Error: No data."
            logger.info(f"Tool returned: {str(_res)[:1000]}")
            return _res
        cat_key = 'category' if 'category' in df.columns else 'Category'
        amt_key = 'amount' if 'amount' in df.columns else 'Amount'
        if category.lower() == "all":
            category_totals = []
            for cat in df[cat_key].unique():
                cat_df = df[df[cat_key] == cat]
                category_totals.append(f"- {cat}: £{cat_df[amt_key].abs().sum():.2f}")
            _res = "Breakdown:\n" + "\n".join(category_totals)
            logger.info(f"Tool returned: {str(_res)[:1000]}")
            return _res
        else:
            cat_df = df[df[cat_key].str.lower() == category.lower()]
            if cat_df.empty:
                _res = f"No transactions for {category}."
                logger.info(f"Tool returned: {str(_res)[:1000]}")
                return _res
            total = cat_df[amt_key].abs().sum()
            _res = f"Total spent in {category}: £{total:.2f}"
            logger.info(f"Tool returned: {str(_res)[:1000]}")
            return _res
    except Exception as e:
        logger.error(f"Error: {e}")
        _res = f"Error: {str(e)}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res

@tool(args_schema=FindHighestSpendingCategoryInput)
def find_highest_spending_category(user_uuid: str, account_ids: list[str], from_date: str = None, to_date: str = None) -> str:
    """
    Identify the single spending category where the user has spent the maximum amount of money.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        account_ids (list[str]): List of account IDs to analyze.
        from_date (str, optional): The start date in YYYY-MM-DD format.
        to_date (str, optional): The end date in YYYY-MM-DD format.
        
    Returns:
        str: A summary of the highest spending category and its total.
    """
    logger.info(f"Executing MCP Tool: find_highest_spending_category")
    try:
        accounts, suffix = _parse_accounts(account_ids, user_uuid)
        df = _get_combined_categorized_data(accounts, suffix, user_uuid, from_date, to_date)
        if df.empty:
            _res = "Error: No data."
            logger.info(f"Tool returned: {str(_res)[:1000]}")
            return _res
        cat_key = 'category' if 'category' in df.columns else 'Category'
        amt_key = 'amount' if 'amount' in df.columns else 'Amount'
        expenses_df = df[df[cat_key].str.lower() != 'income'].copy()
        if expenses_df.empty:
            _res = "No expense categories found."
            logger.info(f"Tool returned: {str(_res)[:1000]}")
            return _res
        grouped = expenses_df.groupby(cat_key)[amt_key].apply(lambda x: x.abs().sum()).reset_index()
        highest = grouped.loc[grouped[amt_key].idxmax()]
        _res = f"Your highest spending category is {highest[cat_key]} with £{highest[amt_key]:.2f}."
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res
    except Exception as e:
        logger.error(f"Error: {e}")
        _res = f"Error: {str(e)}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res

@tool(args_schema=PlotCashFlowMixedInput)
def plot_cash_flow_mixed(user_uuid: str, account_ids: list[str], from_date: str, to_date: str) -> str:
    """
    Generate a cash flow visualization showing daily net income vs expenses.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        account_ids (list[str]): List of account IDs to analyze.
        from_date (str): The start date in YYYY-MM-DD format.
        to_date (str): The end date in YYYY-MM-DD format.
        
    Returns:
        str: A summary text and a chart trigger for the cash flow visualization.
    """
    logger.info(f"Executing MCP Tool: plot_cash_flow_mixed")
    try:
        accounts, _ = _parse_accounts(account_ids, user_uuid)
        from services.api_integrator.account_reader import AccountReader
        payload = []
        data_summary = []
        for acc in accounts:
            user_acc = AccountReader(user_id=user_uuid)
            df = user_acc.get_transactions(acc, user_uuid, from_date, to_date)
            if df.empty:
                continue
            df['date'] = pd.to_datetime(df['date'], format='ISO8601', utc=True)
            df['Month'] = df['date'].dt.strftime('%b %Y')
            monthly = df.groupby('Month').agg(
                Income=('amount', lambda x: x[x > 0].sum()),
                Expense=('amount', lambda x: x[x < 0].abs().sum()),
                Net_Balance=('amount', 'sum'),
                currency=('currency', 'first')
            ).reset_index()
            bank_data = []
            
            acc_total_income = monthly['Income'].sum()
            acc_total_expense = monthly['Expense'].sum()
            data_summary.append(f"- {acc}: Total Income £{acc_total_income:.2f}, Total Expenses £{acc_total_expense:.2f}, Net £{acc_total_income - acc_total_expense:.2f}")
            
            for _, row in monthly.iterrows():
                data_point = {
                    "Month": row['Month'],
                    "Income": round(float(row['Income']), 2),
                    "Expense": round(float(row['Expense']), 2),
                    "Net_Balance": round(float(row['Net_Balance']), 2)
                }
                if 'currency' in row and pd.notna(row['currency']):
                    data_point['currency'] = row['currency']
                bank_data.append(data_point)
            payload.append({"bank_name": acc, "data": bank_data})
            
        if not payload:
            _res = "No cash flow data available for the requested period."
            logger.info(f"Tool returned: {str(_res)[:1000]}")
            return _res
            
        cache_id = _cache_chart_data(payload)
        summary_text = "\n".join(data_summary)
        _res = f"Cash flow chart generated. [TRIGGER_CASH_FLOW_CHART:{cache_id}]\n\nDATA SUMMARY:\n{summary_text}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res
    except Exception as e:
        logger.error(f"Error: {e}")
        _res = f"Error: {str(e)}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res
