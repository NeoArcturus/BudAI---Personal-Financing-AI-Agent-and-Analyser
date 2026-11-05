import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, List, Optional
import pandas as pd
from sqlalchemy import text
from pydantic import BaseModel, Field, model_validator
from config import SessionLocal
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

def _reconcile_account_payload(data: Any) -> Any:
    if isinstance(data, dict):
        if "arguments" in data and isinstance(data["arguments"], dict):
            data = data["arguments"]

        raw_target = data.get("account_id") or data.get("bank_name_or_id") or data.get("account_ids")
        if raw_target is not None:
            if isinstance(raw_target, (list, tuple, set)):
                items = [str(x).strip() for x in raw_target if x]
                first_val = items[0] if items else ""
                data["account_id"] = first_val
                data["account_ids"] = items
            else:
                s_val = str(raw_target).strip()
                data["account_id"] = s_val
                data["account_ids"] = [s_val] if s_val else []
        else:
            data.setdefault("account_id", "")
            data.setdefault("account_ids", [])
    return data

class GenerateFinancialForecastInput(BaseModel):
    user_uuid: str = Field(default="", description="The user UUID.")
    account_id: Optional[str] = Field(default="", description="Target account ID.")
    account_ids: Optional[List[str]] = Field(default_factory=list, description="List of target account IDs.")
    days: int = Field(default=30)
    discipline_multiplier: float = Field(default=1.0)
    drift_adjustment: float = Field(default=0.0)
    stress_test_active: bool = Field(default=False)
    macro_environment: str = Field(default="Stable")

    @model_validator(mode="before")
    @classmethod
    def reconcile_accounts(cls, data: Any) -> Any:
        return _reconcile_account_payload(data)

class GetBudgetVarianceInput(BaseModel):
    user_uuid: str = Field(default="", description="The user UUID.")
    category: str | None = Field(default=None, description="Optional category to filter budget variance for.")

class ForecastBudgetImpactInput(BaseModel):
    user_uuid: str = Field(default="", description="The user UUID.")
    account_id: Optional[str] = Field(default="", description="Target account ID.")
    account_ids: Optional[List[str]] = Field(default_factory=list, description="List of target account IDs.")
    days: int = Field(default=30)

    @model_validator(mode="before")
    @classmethod
    def reconcile_accounts(cls, data: Any) -> Any:
        return _reconcile_account_payload(data)

class GenerateExpenseForecastInput(BaseModel):
    user_uuid: str = Field(default="", description="The user UUID.")
    account_id: Optional[str] = Field(default="", description="Target account ID.")
    account_ids: Optional[List[str]] = Field(default_factory=list, description="List of target account IDs.")
    days: int = Field(default=30)

    @model_validator(mode="before")
    @classmethod
    def reconcile_accounts(cls, data: Any) -> Any:
        return _reconcile_account_payload(data)

class ScenarioInjection(BaseModel):
    day: int = Field(..., description="The day (from today) the event occurs.")
    amount: float = Field(..., description="The financial impact (positive for income, negative for expense).")
    type: str = Field(default="one-off", description="Type of injection: 'one-off' or 'recurring_monthly'.")
    description: str = Field(default="Scenario Event", description="A short description of the event.")

class GenerateHypotheticalScenarioInput(BaseModel):
    user_uuid: str = Field(default="", description="The user UUID.")
    account_id: Optional[str] = Field(default="", description="Target account ID.")
    account_ids: Optional[List[str]] = Field(default_factory=list, description="List of target account IDs.")
    days: int = Field(default=30, description="Forecast horizon in days.")
    injections: List[ScenarioInjection] = Field(default_factory=list, description="List of hypothetical financial events to inject into the forecast.")

    @model_validator(mode="before")
    @classmethod
    def reconcile_accounts(cls, data: Any) -> Any:
        return _reconcile_account_payload(data)

class AnalyzeCriticalSurvivalMetricsInput(BaseModel):
    user_uuid: str = Field(default="", description="The user UUID.")

class AnalyzeWealthAccelerationMetricsInput(BaseModel):
    user_uuid: str = Field(default="", description="The user UUID.")



class PlotHealthRadarInput(BaseModel):
    user_uuid: str = Field(default="", description="The user UUID.")

class UpdateTransactionCategoryInput(BaseModel):
    user_uuid: str = Field(default="", description="The user UUID.")
    transaction_uuid: str = Field(...)
    corrected_category: str = Field(...)

class RetrainCategorizerInput(BaseModel):
    user_uuid: str = Field(default="", description="The user UUID.")

def _cache_chart_data(data: Any) -> str:
    cache_id = f"CACHE_{uuid.uuid4().hex[:8].upper()}"
    try:
        with SessionLocal() as session:
            session.execute(text("INSERT INTO chart_cache (cache_id, chart_data) VALUES (:id, :data)"), {"id": cache_id, "data": json.dumps(data)})
            session.commit()
    except Exception as e:
        logger.error(json.dumps({"message": f"Cache failed: {e}", "status_code": 500}))
        raise
    return cache_id

def _parse_accounts(account_id, user_uuid):
    if not account_id:
        return [], ""
        
    resolved_names = []
    resolved_ids = []
    
    targets = []
    if isinstance(account_id, (list, tuple, set)):
        for item in account_id:
            if isinstance(item, (list, tuple, set)):
                targets.extend([str(x).strip() for x in item if x])
            elif item:
                targets.append(str(item).strip())
    elif account_id:
        targets.append(str(account_id).strip())

    with SessionLocal() as session:
        if not targets:
            rows = session.execute(text("""
                SELECT a.account_id, b.bank_name
                FROM accounts a
                JOIN banks b ON a.bank_uuid = b.bank_uuid
                WHERE a.user_uuid = :user_uuid AND (b.consent_status != '200-403' OR b.consent_status IS NULL)
            """), {"user_uuid": user_uuid}).fetchall()
            for r in rows:
                resolved_ids.append(r[0])
                resolved_names.append(r[1])
        else:
            for ident_clean in targets:
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

    return list(set(resolved_ids)), list(set(resolved_names))

def _get_combined_categorized_data(accounts, suffix, user_uuid, from_date=None, to_date=None):
    combined_df = pd.DataFrame()
    start_date = from_date if from_date else (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    end_date = to_date if to_date else datetime.now().strftime("%Y-%m-%d")
    try:
        with SessionLocal() as session:
            query = """
                SELECT t.transaction_uuid as transaction_id, t.date as timestamp, t.amount, t.description, m.category as "Category", b.bank_name
                FROM transactions t
                LEFT JOIN merchant_knowledge m ON t.merchant_knowledge_uuid = m.knowledge_uuid
                JOIN accounts a ON t.account_id = a.account_id
                JOIN banks b ON a.bank_uuid = b.bank_uuid
                WHERE t.user_uuid = :user_uuid
                AND t.date >= :start_date AND t.date <= :end_date
            """
            params = {"user_uuid": user_uuid, "start_date": start_date, "end_date": end_date}
            
            if accounts:
                query += " AND (b.bank_name = ANY(:accounts) OR a.account_id = ANY(:accounts))"
                params["accounts"] = accounts
            
            query += " ORDER BY t.date DESC"
                
            rows = session.execute(text(query), params).fetchall()
            
            if rows:
                combined_df = pd.DataFrame([dict(row._mapping) for row in rows])
                if not combined_df.empty:
                    combined_df.rename(columns={'timestamp': 'date'}, inplace=True)
                    combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
    except Exception as e:
        logger.error(json.dumps({"message": f"DB query failed: {e}", "status_code": 500}))
    return combined_df
