import json
import logging
import pandas as pd
from langchain_core.tools import tool
from services.mcp_tools.shared_utils import (
    UpdateTransactionCategoryInput, RetrainCategorizerInput,
    _cache_chart_data, _parse_accounts, _get_combined_categorized_data
)
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)


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
    logger.info(json.dumps({"message": f"Executing MCP Tool: update_transaction_category", "status_code": 200}))
    try:
        from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
        agent = CategorizerAgent()
        agent.save_manual_label(user_uuid, transaction_uuid, corrected_category)
        _res = f"Successfully updated transaction {transaction_uuid} to {corrected_category}."
        logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
        return _res
    except Exception as e:
        logger.error(json.dumps({"message": f"Error: {e}", "status_code": 500}))
        _res = f"Error updating: {str(e)}"
        logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
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
    logger.info(json.dumps({"message": f"Executing MCP Tool: retrain_categorization_model", "status_code": 200}))
    try:
        from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
        agent = CategorizerAgent()
        result = agent.retrain_from_feedback(user_uuid)
        if result.get("trained"):
            _res = f"Model successfully retrained using {result.get('samples')} samples."
            logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
            return _res
        else:
            _res = f"Model retraining failed: {result.get('reason')}"
            logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
            return _res
    except Exception as e:
        logger.error(json.dumps({"message": f"Error: {e}", "status_code": 500}))
        _res = f"Error: {str(e)}"
        logger.info(json.dumps({"message": f"Tool returned: {str(_res)[:1000]}", "status_code": 200}))
        return _res
