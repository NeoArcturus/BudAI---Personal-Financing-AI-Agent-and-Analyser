import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any
import pandas as pd
from sqlalchemy import text
from pydantic import BaseModel, Field
from config import SessionLocal
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

# --- SHARED INPUT MODELS ---

class GenerateFinancialForecastInput(BaseModel):
    user_uuid: str = Field(..., description="The unique identifier of the user.")
    bank_name_or_id: str = Field(..., description="The name of the bank or specific account ID.")
    days: int = Field(default=30, description="Number of days to forecast.")
    discipline_multiplier: float = Field(default=1.0, description="Multiplier for spending volatility.")
    drift_adjustment: float = Field(default=0.0, description="Adjustment to the net drift/growth rate.")
    stress_test_active: bool = Field(default=False, description="Whether to simulate a market crash.")
    macro_environment: str = Field(default="Stable", description="The economic environment.")

class ClassifyFinancialDataInput(BaseModel):
    user_uuid: str = Field(..., description="The unique identifier of the user.")
    from_date: str = Field(..., description="Start date in YYYY-MM-DD format.")
    to_date: str = Field(..., description="End date in YYYY-MM-DD format.")
    bank_name_or_id: str = Field(..., description="The name of the bank or specific account ID.")

class FindTotalSpentInput(BaseModel):
    user_uuid: str = Field(..., description="The unique identifier of the user.")
    category_name: str = Field(..., description="The category to search for. Use 'ALL' for a full breakdown.")
    bank_name_or_id: str = Field(..., description="The name of the bank or specific account ID.")

class FindHighestSpendingCategoryInput(BaseModel):
    user_uuid: str = Field(..., description="The unique identifier of the user.")
    bank_name_or_id: str = Field(..., description="The name of the bank or specific account ID.")

class CreateBargraphChartInput(BaseModel):
    user_uuid: str = Field(..., description="The unique identifier of the user.")
    bank_name_or_id: str = Field(..., description="The name of the bank or specific account ID.")

class CreatePieChartInput(BaseModel):
    user_uuid: str = Field(..., description="The unique identifier of the user.")
    bank_name_or_id: str = Field(..., description="The name of the bank or specific account ID.")

class PlotExpensesInput(BaseModel):
    user_uuid: str = Field(..., description="The unique identifier of the user.")
    plot_time_type: str = Field(..., description="The frequency: 'Daily', 'Weekly', or 'Monthly'.")
    from_date: str = Field(..., description="Start date in YYYY-MM-DD format.")
    to_date: str = Field(..., description="End date in YYYY-MM-DD format.")
    bank_name_or_id: str = Field(..., description="The name of the bank or specific account ID.")

class GenerateExpenseForecastInput(BaseModel):
    user_uuid: str = Field(..., description="The unique identifier of the user.")
    bank_name_or_id: str = Field(..., description="The name of the bank or specific account ID.")
    days: int = Field(default=30, description="Number of days to forecast.")

class AnalyzeCriticalSurvivalMetricsInput(BaseModel):
    user_uuid: str = Field(..., description="The unique identifier of the user.")
    bank_name_or_id: str = Field(..., description="The name of the bank or specific account ID.")

class AnalyzeWealthAccelerationMetricsInput(BaseModel):
    user_uuid: str = Field(..., description="The unique identifier of the user.")
    bank_name_or_id: str = Field(..., description="The name of the bank or specific account ID.")

class PlotCashFlowMixedInput(BaseModel):
    user_uuid: str = Field(..., description="The unique identifier of the user.")
    bank_name_or_id: str = Field(..., description="The name of the bank or specific account ID.")
    from_date: str = Field(..., description="Start date in YYYY-MM-DD format.")
    to_date: str = Field(..., description="End date in YYYY-MM-DD format.")

class PlotHealthRadarInput(BaseModel):
    user_uuid: str = Field(..., description="The unique identifier of the user.")
    bank_name_or_id: str = Field(..., description="The name of the bank or specific account ID.")

class UpdateTransactionCategoryInput(BaseModel):
    user_uuid: str = Field(..., description="The unique identifier of the user.")
    transaction_uuid: str = Field(..., description="The unique ID of the transaction to update.")
    corrected_label: str = Field(..., description="The new category label to assign.")

class RetrainCategorizerInput(BaseModel):
    user_uuid: str = Field(..., description="The unique identifier of the user.")

# --- SHARED UTILITIES ---

def _cache_chart_data(data: Any) -> str:
    logger.debug("Caching chart data...")
    cache_id = f"CACHE_{uuid.uuid4().hex[:8].upper()}"
    try:
        with SessionLocal() as session:
            session.execute(
                text("CREATE TABLE IF NOT EXISTS chart_cache (cache_id TEXT PRIMARY KEY, chart_data TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            )
            session.execute(
                text("INSERT INTO chart_cache (cache_id, chart_data) VALUES (:id, :data)"),
                {"id": cache_id, "data": json.dumps(data)}
            )
            session.commit()
        logger.info(f"Chart data cached successfully with ID: {cache_id}")
    except Exception as e:
        logger.error(f"Failed to cache chart data: {e}", exc_info=True)
        raise
    return cache_id

def _parse_accounts(bank_name_or_id, user_uuid):
    logger.debug(f"Parsing accounts in internal_tools for identifier: {bank_name_or_id}")
    if not bank_name_or_id or str(bank_name_or_id).lower() in ["none", ""]:
        return [], ""

    identifiers = []
    if isinstance(bank_name_or_id, str):
        identifiers = [i.strip().replace("'", "").replace("\’", "") for i in bank_name_or_id.split(",") if i.strip()]
    elif isinstance(bank_name_or_id, list):
        identifiers = [str(i).strip().replace("'", "").replace("\’", "") for i in bank_name_or_id if str(i).strip()]

    if not identifiers:
        return [], ""

    if "ALL" in [i.upper() for i in identifiers]:
        with SessionLocal() as session:
            accounts = [row[0] for row in session.execute(
                text("SELECT b.bank_name FROM banks b WHERE b.user_uuid = :user_uuid"), {"user_uuid": user_uuid}).fetchall()]
        return accounts, "ALL"

    resolved_names = []
    first_resolved_id = identifiers[0]
    with SessionLocal() as session:
        for ident in identifiers:
            row = session.execute(text("""
                SELECT a.account_id, b.bank_name
                FROM accounts a
                JOIN banks b ON a.bank_uuid = b.bank_uuid
                WHERE (b.bank_name = :ident OR a.account_id = :ident) AND a.user_uuid = :user_uuid
            """), {"ident": ident, "user_uuid": user_uuid}).fetchone()
            if row:
                resolved_names.append(row[1])
                if ident == identifiers[0]:
                    first_resolved_id = row[0]
            else:
                resolved_names.append(ident)
    return list(set(resolved_names)), first_resolved_id if len(identifiers) == 1 else ",".join(identifiers)

def _get_combined_categorized_data(accounts, suffix, user_uuid):
    logger.info(f"Gathering combined categorized data for user: {user_uuid}")
    combined_df = pd.DataFrame()
    from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
    agent = CategorizerAgent()
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    for acc in accounts:
        try:
            logger.debug(f"Executing categorization cycle for account: {acc}")
            df = agent.execute_cycle(acc, user_uuid, start_date, end_date)
            if df is not None and not df.empty:
                df['bank_name'] = acc
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                logger.debug(f"Fetched {len(df)} transactions for account: {acc}")
        except Exception as e:
            logger.error(f"Error categorizing account {acc}: {e}")
            pass
    logger.info(f"Combined data total rows: {len(combined_df)}")
    return combined_df
