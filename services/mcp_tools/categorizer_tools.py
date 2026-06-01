import logging
import pandas as pd
from langchain_core.tools import tool
from services.mcp_tools.shared_utils import (
    ClassifyFinancialDataInput, CreateBargraphChartInput, CreatePieChartInput,
    UpdateTransactionCategoryInput, RetrainCategorizerInput,
    _cache_chart_data, _parse_accounts, _get_combined_categorized_data
)
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

@tool(args_schema=ClassifyFinancialDataInput)
def classify_financial_data(user_uuid: str, from_date: str, to_date: str, account_ids: list[str]) -> str:
    """Categorize and classify the user's raw bank transactions into distinct spending categories."""
    try:
        accounts, suffix = _parse_accounts(account_ids, user_uuid)
        from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
        agent = CategorizerAgent()
        combined_df = pd.DataFrame()
        for acc in accounts:
            df = agent.execute_cycle(acc, user_uuid, from_date, to_date)
            if df is not None and not df.empty:
                df['bank_name'] = acc
                combined_df = pd.concat([combined_df, df], ignore_index=True)
        if combined_df.empty:
            return "No transactions found."
        payload = []
        for acc in accounts:
            acc_df = combined_df[combined_df['bank_name'] == acc]
            bank_data = []
            cat_key = 'category' if 'category' in acc_df.columns else 'Category'
            amt_key = 'amount' if 'amount' in acc_df.columns else 'Amount'
            for cat in acc_df[cat_key].unique():
                cat_df = acc_df[acc_df[cat_key] == cat]
                bank_data.append({
                    "Category": str(cat),
                    "Total_Amount": round(float(cat_df[amt_key].abs().sum()), 2),
                    "count": len(cat_df)
                })
            bank_data.sort(key=lambda x: x["Total_Amount"], reverse=True)
            payload.append({"bank_name": acc, "data": bank_data})
        cache_id = _cache_chart_data(payload)
        return f"Classified {len(combined_df)} transactions. [TRIGGER_CATEGORIZED_CHART:{cache_id}]"
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Error: {str(e)}"

@tool(args_schema=CreateBargraphChartInput)
def create_bargraph_chart_and_save(user_uuid: str, account_ids: list[str]) -> str:
    """Generate a visual distribution chart of the user's categorized spending."""
    try:
        accounts, suffix = _parse_accounts(account_ids, user_uuid)
        df = _get_combined_categorized_data(accounts, suffix, user_uuid)
        if df.empty:
            return "No data to chart."
        payload = []
        cat_key = 'category' if 'category' in df.columns else 'Category'
        amt_key = 'amount' if 'amount' in df.columns else 'Amount'
        for acc in accounts:
            acc_df = df[df['bank_name'] == acc]
            bank_data = []
            for cat in acc_df[cat_key].unique():
                cat_df = acc_df[acc_df[cat_key] == cat]
                bank_data.append({
                    "Category": str(cat),
                    "Total_Amount": round(float(cat_df[amt_key].abs().sum()), 2),
                    "count": len(cat_df)
                })
            bank_data.sort(key=lambda x: x["Total_Amount"], reverse=True)
            payload.append({"bank_name": acc, "data": bank_data})
        cache_id = _cache_chart_data(payload)
        return f"Chart generated. [TRIGGER_CATEGORIZED_CHART:{cache_id}]"
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Error: {str(e)}"

@tool(args_schema=CreatePieChartInput)
def create_pie_chart_and_save(user_uuid: str, account_ids: list[str]) -> str:
    """Generate a visual pie/doughnut chart representing the proportional distribution of the user's categorized spending."""
    try:
        accounts, suffix = _parse_accounts(account_ids, user_uuid)
        df = _get_combined_categorized_data(accounts, suffix, user_uuid)
        if df.empty:
            return "No data to chart."
        payload = []
        cat_key = 'category' if 'category' in df.columns else 'Category'
        amt_key = 'amount' if 'amount' in df.columns else 'Amount'
        for acc in accounts:
            acc_df = df[df['bank_name'] == acc]
            bank_data = []
            for cat in acc_df[cat_key].unique():
                cat_df = acc_df[acc_df[cat_key] == cat]
                bank_data.append({
                    "Category": str(cat),
                    "Total_Amount": round(float(cat_df[amt_key].abs().sum()), 2),
                    "count": len(cat_df)
                })
            bank_data.sort(key=lambda x: x["Total_Amount"], reverse=True)
            payload.append({"bank_name": acc, "data": bank_data})
        cache_id = _cache_chart_data(payload)
        return f"Chart generated. [TRIGGER_CATEGORIZED_DOUGHNUT_CHART:{cache_id}]"
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Error: {str(e)}"

@tool(args_schema=UpdateTransactionCategoryInput)
def update_transaction_category(user_uuid: str, transaction_uuid: str, corrected_category: str) -> str:
    """Manually update the category of a specific transaction and trigger feedback learning."""
    try:
        from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
        agent = CategorizerAgent()
        agent.save_manual_label(user_uuid, transaction_uuid, corrected_category)
        return f"Successfully updated transaction {transaction_uuid} to {corrected_category}."
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Error updating: {str(e)}"

@tool(args_schema=RetrainCategorizerInput)
def retrain_categorization_model(user_uuid: str) -> str:
    """Trigger the machine learning model to retrain based on all corrected manual feedback provided so far."""
    try:
        from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
        agent = CategorizerAgent()
        result = agent.retrain_from_feedback(user_uuid)
        if result.get("trained"):
            return f"Model successfully retrained using {result.get('samples')} samples."
        else:
            return f"Model retraining failed: {result.get('reason')}"
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Error: {str(e)}"
