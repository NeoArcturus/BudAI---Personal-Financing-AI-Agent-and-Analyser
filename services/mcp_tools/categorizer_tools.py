import logging
import pandas as pd
from langchain_core.tools import tool
from services.mcp_tools.shared_utils import (
    CreateBargraphChartInput, CreatePieChartInput,
    UpdateTransactionCategoryInput, RetrainCategorizerInput,
    _cache_chart_data, _parse_accounts, _get_combined_categorized_data
)
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)


@tool(args_schema=CreateBargraphChartInput)
def create_bargraph_chart_and_save(user_uuid: str, account_ids: list[str]) -> str:
    """
    Generate a visual distribution chart of the user's categorized spending.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        account_ids (list[str]): List of account IDs to chart.
        
    Returns:
        str: A summary of the categorical spending and a chart trigger.
    """
    logger.info(f"Executing MCP Tool: create_bargraph_chart_and_save")
    try:
        accounts, suffix = _parse_accounts(account_ids, user_uuid)
        df = _get_combined_categorized_data(accounts, suffix, user_uuid)
        if df.empty:
            _res = "No data to chart."
            logger.info(f"Tool returned: {str(_res)[:1000]}")
            return _res
        payload = []
        data_summary = []
        cat_key = 'category' if 'category' in df.columns else 'Category'
        amt_key = 'amount' if 'amount' in df.columns else 'Amount'
        for acc in accounts:
            acc_df = df[df['bank_name'] == acc]
            if acc_df.empty: continue
            
            bank_data = []
            acc_summary = [f"Account: {acc}"]
            for cat in acc_df[cat_key].unique():
                cat_df = acc_df[acc_df[cat_key] == cat]
                total = round(float(cat_df[amt_key].abs().sum()), 2)
                
                currency = None
                if 'currency' in cat_df.columns and not cat_df['currency'].empty:
                    currency = cat_df['currency'].iloc[0]
                    
                data_point = {
                    "Category": str(cat),
                    "Total_Amount": total,
                    "count": len(cat_df)
                }
                if currency and pd.notna(currency):
                    data_point['currency'] = currency
                    
                bank_data.append(data_point)
                acc_summary.append(f"- {cat}: £{total}")
            
            data_summary.append("\n".join(acc_summary))
            bank_data.sort(key=lambda x: x["Total_Amount"], reverse=True)
            payload.append({"bank_name": acc, "data": bank_data})
            
        cache_id = _cache_chart_data(payload)
        summary_text = "\n\n".join(data_summary)
        _res = f"Chart generated. [TRIGGER_CATEGORIZED_CHART:{cache_id}]\n\nDATA SUMMARY:\n{summary_text}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res
    except Exception as e:
        logger.error(f"Error: {e}")
        _res = f"Error: {str(e)}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res

@tool(args_schema=CreatePieChartInput)
def create_pie_chart_and_save(user_uuid: str, account_ids: list[str]) -> str:
    """
    Generate a visual pie/doughnut chart representing the proportional distribution of the user's categorized spending.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        account_ids (list[str]): List of account IDs to chart.
        
    Returns:
        str: A summary of the proportional spending and a chart trigger.
    """
    logger.info(f"Executing MCP Tool: create_pie_chart_and_save")
    try:
        accounts, suffix = _parse_accounts(account_ids, user_uuid)
        df = _get_combined_categorized_data(accounts, suffix, user_uuid)
        if df.empty:
            _res = "No data to chart."
            logger.info(f"Tool returned: {str(_res)[:1000]}")
            return _res
        payload = []
        data_summary = []
        cat_key = 'category' if 'category' in df.columns else 'Category'
        amt_key = 'amount' if 'amount' in df.columns else 'Amount'
        for acc in accounts:
            acc_df = df[df['bank_name'] == acc]
            if acc_df.empty: continue
            
            bank_data = []
            acc_summary = [f"Account: {acc}"]
            for cat in acc_df[cat_key].unique():
                cat_df = acc_df[acc_df[cat_key] == cat]
                total = round(float(cat_df[amt_key].abs().sum()), 2)
                
                currency = None
                if 'currency' in cat_df.columns and not cat_df['currency'].empty:
                    currency = cat_df['currency'].iloc[0]
                    
                data_point = {
                    "Category": str(cat),
                    "Total_Amount": total,
                    "count": len(cat_df)
                }
                if currency and pd.notna(currency):
                    data_point['currency'] = currency
                    
                bank_data.append(data_point)
                acc_summary.append(f"- {cat}: £{total}")
            
            data_summary.append("\n".join(acc_summary))
            bank_data.sort(key=lambda x: x["Total_Amount"], reverse=True)
            payload.append({"bank_name": acc, "data": bank_data})
            
        cache_id = _cache_chart_data(payload)
        summary_text = "\n\n".join(data_summary)
        _res = f"Chart generated. [TRIGGER_CATEGORIZED_DOUGHNUT_CHART:{cache_id}]\n\nDATA SUMMARY:\n{summary_text}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res
    except Exception as e:
        logger.error(f"Error: {e}")
        _res = f"Error: {str(e)}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res

@tool(args_schema=UpdateTransactionCategoryInput)
def update_transaction_category(user_uuid: str, transaction_uuid: str, corrected_category: str) -> str:
    """
    Manually update the category of a specific transaction and trigger feedback learning.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        transaction_uuid (str): The transaction ID to update.
        corrected_category (str): The new category label.
        
    Returns:
        str: A success or error message for the update operation.
    """
    logger.info(f"Executing MCP Tool: update_transaction_category")
    try:
        from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
        agent = CategorizerAgent()
        agent.save_manual_label(user_uuid, transaction_uuid, corrected_category)
        _res = f"Successfully updated transaction {transaction_uuid} to {corrected_category}."
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res
    except Exception as e:
        logger.error(f"Error: {e}")
        _res = f"Error updating: {str(e)}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res

@tool(args_schema=RetrainCategorizerInput)
def retrain_categorization_model(user_uuid: str) -> str:
    """
    Trigger the machine learning model to retrain based on all corrected manual feedback provided so far.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        
    Returns:
        str: A status message detailing the success or failure of the retraining.
    """
    logger.info(f"Executing MCP Tool: retrain_categorization_model")
    try:
        from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
        agent = CategorizerAgent()
        result = agent.retrain_from_feedback(user_uuid)
        if result.get("trained"):
            _res = f"Model successfully retrained using {result.get('samples')} samples."
            logger.info(f"Tool returned: {str(_res)[:1000]}")
            return _res
        else:
            _res = f"Model retraining failed: {result.get('reason')}"
            logger.info(f"Tool returned: {str(_res)[:1000]}")
            return _res
    except Exception as e:
        logger.error(f"Error: {e}")
        _res = f"Error: {str(e)}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res
