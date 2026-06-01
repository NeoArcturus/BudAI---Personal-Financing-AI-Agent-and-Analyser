from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Any, Dict
import os
import json
import uuid
import pandas as pd
from sqlalchemy import text
from datetime import datetime
from cryptography.fernet import Fernet
from config import SessionLocal
from services.api_integrator.access_token_generator import AccessTokenGenerator
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class BaseToolInput(BaseModel):
    @model_validator(mode='before')
    @classmethod
    def unnest_args(cls, data: object) -> object:
        if isinstance(data, dict):
            if 'arguments' in data and isinstance(data['arguments'], dict):
                return data['arguments']
        return data

class GenerateFinancialForecastInput(BaseToolInput):
    account_ids: List[str] = Field(..., description="List of account UUIDs or names.")
    user_uuid: str = Field(..., description="The exact user_uuid string.")
    days: int = Field(60, description="Days to forecast.")
    discipline_multiplier: float = Field(1.0, description="Spending discipline multiplier.")
    drift_adjustment: float = Field(0.0, description="Growth drift adjustment.")
    stress_test_active: bool = Field(False, description="Simulate stress test.")
    macro_environment: str = Field("Stable", description="Economic environment.")

class ClassifyFinancialDataInput(BaseToolInput):
    from_date: str = Field(..., description="Start date YYYY-MM-DD.")
    to_date: str = Field(..., description="End date YYYY-MM-DD.")
    account_ids: List[str] = Field(..., description="List of account UUIDs or names.")
    user_uuid: str = Field(..., description="The exact user_uuid string.")

class FindTotalSpentInput(BaseToolInput):
    category_name: str = Field(..., description="Category name or 'all'.")
    account_ids: List[str] = Field(..., description="List of account UUIDs or names.")
    user_uuid: str = Field(..., description="The exact user_uuid string.")

class FindHighestSpendingCategoryInput(BaseToolInput):
    account_ids: List[str] = Field(..., description="List of account UUIDs or names.")
    user_uuid: str = Field(..., description="The exact user_uuid string.")

class CreateBargraphChartInput(BaseToolInput):
    account_ids: List[str] = Field(..., description="List of account UUIDs or names.")
    user_uuid: str = Field(..., description="The exact user_uuid string.")

class CreatePieChartInput(BaseToolInput):
    account_ids: List[str] = Field(..., description="List of account UUIDs or names.")
    user_uuid: str = Field(..., description="The exact user_uuid string.")

class PlotExpensesInput(BaseToolInput):
    plot_time_type: str = Field(..., description="'Daily', 'Weekly', or 'Monthly'.")
    from_date: str = Field(..., description="Start date YYYY-MM-DD.")
    to_date: str = Field(..., description="End date YYYY-MM-DD.")
    account_ids: List[str] = Field(..., description="List of account UUIDs or names.")
    user_uuid: str = Field(..., description="The exact user_uuid string.")

class GenerateExpenseForecastInput(BaseToolInput):
    account_ids: List[str] = Field(..., description="List of account UUIDs or names.")
    user_uuid: str = Field(..., description="The exact user_uuid string.")
    days: int = Field(30, description="Days to forecast.")
    discipline_multiplier: float = Field(1.0, description="Discipline multiplier.")
    drift_adjustment: float = Field(0.0, description="Drift adjustment.")
    stress_test_active: bool = Field(False, description="Simulate stress test.")
    macro_environment: str = Field("Stable", description="Economic environment.")

class AnalyzeCriticalSurvivalMetricsInput(BaseToolInput):
    user_uuid: str = Field(..., description="The exact user_uuid string.")

class AnalyzeWealthAccelerationMetricsInput(BaseToolInput):
    user_uuid: str = Field(..., description="The exact user_uuid string.")

class PlotCashFlowMixedInput(BaseToolInput):
    account_ids: List[str] = Field(..., description="List of account UUIDs or names.")
    user_uuid: str = Field(..., description="The exact user_uuid string.")
    from_date: str = Field(...)
    to_date: str = Field(...)

class PlotHealthRadarInput(BaseToolInput):
    user_uuid: str = Field(..., description="The exact user_uuid string.")

class UpdateTransactionCategoryInput(BaseToolInput):
    user_uuid: str = Field(..., description="The exact user_uuid string.")
    transaction_uuid: str = Field(..., description="Unique transaction ID.")
    corrected_category: str = Field(..., description="New category label.")

class RetrainCategorizerInput(BaseToolInput):
    user_uuid: str = Field(..., description="The exact user_uuid string.")

class ExportAdvisoryStateInput(BaseToolInput):
    user_uuid: str = Field(..., description="The user UUID.")
    chart_type: str = Field(..., description="Chart type.")
    raw_data: dict = Field(..., description="Raw data payload.")
    ai_analysis: str = Field(..., description="Analytical text.")

class ExportAnalyzedStatementInput(BaseModel):
    user_uuid: str = Field(..., description="The alphanumeric user UUID string.")
    ai_summary: str = Field(..., description="A short summary.")

class MemorySearchInput(BaseModel):
    query: str = Field(..., description="Search concept.")

class MemoryExtractionInput(BaseModel):
    entities: list = Field(..., description="Entity list.")

def _cache_chart_data(data: Any) -> str:
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
    except Exception as e:
        logger.error(f"Cache failed: {e}")
        raise
    return cache_id

def _check_and_handle_sca_error(e, acc, user_uuid):
    if isinstance(e, PermissionError) and "SECURITY LOCK" in str(e):
        with SessionLocal() as session:
            row = session.execute(text("""
                SELECT b.truelayer_provider_id, b.refresh_token
                FROM banks b
                LEFT JOIN accounts a ON a.bank_uuid = b.bank_uuid
                WHERE (b.bank_name = :acc OR a.account_id = :acc) AND b.user_uuid = :user_uuid
            """), {"acc": acc, "user_uuid": user_uuid}).fetchone()
        token_gen = AccessTokenGenerator()
        if row and row[1]:
            provider_id = row[0]
            enc_key = os.getenv("ENCRYPTION_KEY", b'cw_8H_1M4bX_3nF8vO5n3Y7A8xQ3_1m8aT2vP5_v5r8=')
            cipher_suite = Fernet(enc_key)
            refresh_token = cipher_suite.decrypt(bytes(row[1])).decode()
            auth_link = token_gen.get_reauth_link(refresh_token, user_uuid)
            if not auth_link:
                auth_link = token_gen.get_auth_link(user_uuid)
                auth_link += f"&provider_id={provider_id}"
        else:
            auth_link = token_gen.get_auth_link(user_uuid)
        return (f"⚠️ **SCA Security Lock Activated**\n\n"
                f"Under Open Banking regulations, {acc} requires you to periodically re-authenticate.\n\n"
                f"**[Click here to securely re-authenticate with {acc}]({auth_link})**")
    return None

def _parse_accounts(account_ids, user_uuid):
    if not account_ids:
        return [], ""
    resolved_names = []
    resolved_ids = []
    with SessionLocal() as session:
        for ident in account_ids:
            row = session.execute(text("""
                SELECT a.account_id, b.bank_name
                FROM accounts a
                JOIN banks b ON a.bank_uuid = b.bank_uuid
                WHERE (b.bank_name = :ident OR a.account_id = :ident) AND a.user_uuid = :user_uuid
            """), {"ident": ident, "user_uuid": user_uuid}).fetchone()
            if row:
                resolved_ids.append(row[0])
                resolved_names.append(row[1])
            else:
                resolved_ids.append(ident)
                resolved_names.append(ident)
    return list(set(resolved_names)), ",".join(list(set(resolved_ids)))

def _get_combined_categorized_data(accounts, suffix, user_uuid):
    from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
    combined_df = pd.DataFrame()
    agent = CategorizerAgent()
    start_date = (datetime.now() - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    for acc in accounts:
        try:
            df = agent.execute_cycle(acc, user_uuid, start_date, end_date)
            if df is not None and not df.empty:
                df['bank_name'] = acc
                combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception:
            pass
    return combined_df
