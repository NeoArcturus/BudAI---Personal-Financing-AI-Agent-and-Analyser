import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, List
import pandas as pd
from sqlalchemy import text
from pydantic import BaseModel, Field
from config import SessionLocal
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class GenerateFinancialForecastInput(BaseModel):
    user_uuid: str = Field(..., description="The user UUID.")
    account_ids: List[str] = Field(..., description="List of accounts.")
    days: int = Field(default=30)
    discipline_multiplier: float = Field(default=1.0)
    drift_adjustment: float = Field(default=0.0)
    stress_test_active: bool = Field(default=False)
    macro_environment: str = Field(default="Stable")

class ClassifyFinancialDataInput(BaseModel):
    user_uuid: str = Field(..., description="The user UUID.")
    from_date: str = Field(...)
    to_date: str = Field(...)
    account_ids: List[str] = Field(...)

class FindTotalSpentInput(BaseModel):
    user_uuid: str = Field(..., description="The user UUID.")
    category_name: str = Field(...)
    account_ids: List[str] = Field(...)
    from_date: str | None = Field(default=None)
    to_date: str | None = Field(default=None)

class FindHighestSpendingCategoryInput(BaseModel):
    user_uuid: str = Field(..., description="The user UUID.")
    account_ids: List[str] = Field(...)
    from_date: str | None = Field(default=None)
    to_date: str | None = Field(default=None)

class CreateBargraphChartInput(BaseModel):
    user_uuid: str = Field(..., description="The user UUID.")
    account_ids: List[str] = Field(...)

class CreatePieChartInput(BaseModel):
    user_uuid: str = Field(..., description="The user UUID.")
    account_ids: List[str] = Field(...)

class PlotExpensesInput(BaseModel):
    user_uuid: str = Field(..., description="The user UUID.")
    plot_time_type: str = Field(...)
    from_date: str = Field(...)
    to_date: str = Field(...)
    account_ids: List[str] = Field(...)

class GenerateExpenseForecastInput(BaseModel):
    user_uuid: str = Field(..., description="The user UUID.")
    account_ids: List[str] = Field(...)
    days: int = Field(default=30)

class AnalyzeCriticalSurvivalMetricsInput(BaseModel):
    user_uuid: str = Field(..., description="The user UUID.")

class AnalyzeWealthAccelerationMetricsInput(BaseModel):
    user_uuid: str = Field(..., description="The user UUID.")

class PlotCashFlowMixedInput(BaseModel):
    user_uuid: str = Field(..., description="The user UUID.")
    account_ids: List[str] = Field(...)
    from_date: str = Field(...)
    to_date: str = Field(...)

class PlotHealthRadarInput(BaseModel):
    user_uuid: str = Field(..., description="The user UUID.")

class UpdateTransactionCategoryInput(BaseModel):
    user_uuid: str = Field(..., description="The user UUID.")
    transaction_uuid: str = Field(...)
    corrected_category: str = Field(...)

class RetrainCategorizerInput(BaseModel):
    user_uuid: str = Field(..., description="The user UUID.")

def _cache_chart_data(data: Any) -> str:
    cache_id = f"CACHE_{uuid.uuid4().hex[:8].upper()}"
    try:
        with SessionLocal() as session:
            session.execute(text("INSERT INTO chart_cache (cache_id, chart_data) VALUES (:id, :data)"), {"id": cache_id, "data": json.dumps(data)})
            session.commit()
    except Exception as e:
        logger.error(f"Cache failed: {e}")
        raise
    return cache_id

def _parse_accounts(account_ids, user_uuid):
    if not account_ids:
        return [], ""
    
    if isinstance(account_ids, str):
        account_ids = [account_ids]
        
    resolved_names = []
    resolved_ids = []
    
    with SessionLocal() as session:
        if any(str(i).strip().upper() == "ALL" for i in account_ids):
            rows = session.execute(text("""
                SELECT a.account_id, b.bank_name
                FROM accounts a
                JOIN banks b ON a.bank_uuid = b.bank_uuid
                WHERE a.user_uuid = :user_uuid AND (b.consent_status != 'revoked' OR b.consent_status IS NULL)
            """), {"user_uuid": user_uuid}).fetchall()
            for r in rows:
                resolved_ids.append(r[0])
                resolved_names.append(r[1])
        else:
            for ident in account_ids:
                ident_clean = str(ident).strip()
                row = session.execute(text("""
                    SELECT a.account_id, b.bank_name
                    FROM accounts a
                    JOIN banks b ON a.bank_uuid = b.bank_uuid
                    WHERE (b.bank_name ILIKE :ident OR a.account_id = :ident_exact) AND a.user_uuid = :user_uuid
                """), {"ident": ident_clean, "ident_exact": ident_clean, "user_uuid": user_uuid}).fetchone()

                if row:
                    resolved_ids.append(row[0])
                    resolved_names.append(row[1])
                else:
                    resolved_ids.append(ident_clean)
                    resolved_names.append(ident_clean)

    return list(set(resolved_names)), ",".join(list(set(resolved_ids)))

def _get_combined_categorized_data(accounts, suffix, user_uuid, from_date=None, to_date=None):
    combined_df = pd.DataFrame()
    start_date = from_date if from_date else (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    end_date = to_date if to_date else datetime.now().strftime("%Y-%m-%d")
    try:
        with SessionLocal() as session:
            query = """
                SELECT t.transaction_uuid as transaction_id, t.date as timestamp, t.amount, t.description, t.category as "Category", b.bank_name
                FROM transactions t
                JOIN accounts a ON t.account_id = a.account_id
                JOIN banks b ON a.bank_uuid = b.bank_uuid
                WHERE t.user_uuid = :user_uuid
                AND t.date >= :start_date AND t.date <= :end_date
            """
            params = {"user_uuid": user_uuid, "start_date": start_date, "end_date": end_date}
            
            if accounts and "ALL" not in [str(acc).upper() for acc in accounts]:
                query += " AND (b.bank_name = ANY(:accounts) OR a.account_id = ANY(:accounts))"
                params["accounts"] = accounts
                
            rows = session.execute(text(query), params).fetchall()
            
            if rows:
                combined_df = pd.DataFrame([dict(row._mapping) for row in rows])
                if not combined_df.empty:
                    combined_df.rename(columns={'timestamp': 'date'}, inplace=True)
                    combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
    except Exception as e:
        logger.error(f"DB query failed: {e}")
    return combined_df

