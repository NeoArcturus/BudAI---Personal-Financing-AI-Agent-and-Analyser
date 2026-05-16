import os
import json
import uuid
import pandas as pd
from sqlalchemy import text
from datetime import datetime
from cryptography.fernet import Fernet
from pydantic import BaseModel, Field, model_validator
from config import SessionLocal
from services.api_integrator.access_token_generator import AccessTokenGenerator


class BaseToolInput(BaseModel):
    @model_validator(mode='before')
    @classmethod
    def unnest_args(cls, data: object) -> object:
        if isinstance(data, dict):
            if 'arguments' in data and isinstance(data['arguments'], dict):
                return data['arguments']
        return data


class GenerateFinancialForecastInput(BaseToolInput):
    bank_name_or_id: str = Field(...,
                                 description="MUST be 'ALL' unless explicitly named.")
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")
    days: int = Field(
        60, description="The number of days into the future to forecast.")
    discipline_multiplier: float = Field(
        1.0, description="Multiplier for spending discipline (Strict < 1.0, Erratic > 1.0).")
    drift_adjustment: float = Field(
        0.0, description="Adjustment to the baseline growth drift.")
    stress_test_active: bool = Field(
        False, description="Whether to simulate a major negative life event.")
    macro_environment: str = Field(
        "Stable", description="Economic environment: 'Stable', 'Inflationary', or 'Recession'.")


class ClassifyFinancialDataInput(BaseToolInput):
    from_date: str = Field(...,
                           description="Starting date for transactions in YYYY-MM-DD format.")
    to_date: str = Field(...,
                         description="End date for transactions in YYYY-MM-DD format.")
    bank_name_or_id: str = Field(...,
                                 description="MUST be 'ALL' unless explicitly named.")
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


class FindTotalSpentInput(BaseToolInput):
    category_name: str = Field(...,
                               description="Name of the category, or 'all'.")
    bank_name_or_id: str = Field(...,
                                 description="MUST be 'ALL' unless explicitly named.")
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


class FindHighestSpendingCategoryInput(BaseToolInput):
    bank_name_or_id: str = Field(...,
                                 description="MUST be 'ALL' unless explicitly named.")
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


class CreateBargraphChartInput(BaseToolInput):
    bank_name_or_id: str = Field(...,
                                 description="MUST be 'ALL' unless explicitly named.")
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


class CreatePieChartInput(BaseToolInput):
    bank_name_or_id: str = Field(...,
                                 description="MUST be 'ALL' unless explicitly named.")
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


class PlotExpensesInput(BaseToolInput):
    plot_time_type: str = Field(...,
                                description="MUST be exactly 'Daily', 'Weekly', or 'Monthly'.")
    from_date: str = Field(...,
                           description="Starting date for transactions in YYYY-MM-DD format.")
    to_date: str = Field(...,
                         description="End date for transactions in YYYY-MM-DD format.")
    bank_name_or_id: str = Field(...,
                                 description="MUST be 'ALL' unless explicitly named.")
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


class GenerateExpenseForecastInput(BaseToolInput):
    bank_name_or_id: str = Field(...,
                                 description="MUST be 'ALL' unless explicitly named.")
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")
    days: int = Field(
        30, description="The number of days into the future to forecast.")
    discipline_multiplier: float = Field(
        1.0, description="Multiplier for spending discipline.")
    drift_adjustment: float = Field(
        0.0, description="Adjustment to the baseline drift.")
    stress_test_active: bool = Field(
        False, description="Whether to simulate a stress test.")
    macro_environment: str = Field(
        "Stable", description="Economic environment.")


class AnalyzeCriticalSurvivalMetricsInput(BaseToolInput):
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


class AnalyzeWealthAccelerationMetricsInput(BaseToolInput):
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


class PlotCashFlowMixedInput(BaseToolInput):
    bank_name_or_id: str = Field(...,
                                 description="MUST be 'ALL' unless explicitly named.")
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


class PlotHealthRadarInput(BaseToolInput):
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


class UpdateTransactionCategoryInput(BaseToolInput):
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")
    transaction_uuid: str = Field(
        ..., description="The unique ID of the transaction to be updated.")
    corrected_category: str = Field(
        ..., description="The new, user-verified category label (e.g., 'Food & Dining').")


class RetrainCategorizerInput(BaseToolInput):
    user_uuid: str = Field(
        ..., description="The exact alphanumeric user_uuid string. DO NOT use placeholders.")


class ExportAdvisoryStateInput(BaseToolInput):
    user_uuid: str = Field(...,
                           description="The exact alphanumeric user_uuid string.")
    chart_type: str = Field(..., description="The type of chart analyzed.")
    raw_data: dict = Field(..., description="The raw chart data payload.")
    ai_analysis: str = Field(..., description="The finalized analytical text.")


class ExportAnalyzedStatementInput(BaseModel):
    user_uuid: str = Field(...,
                           description="The alphanumeric user UUID string.")
    ai_summary: str = Field(
        ..., description="A short summary of the user's financial health to inject into the file.")


class MemorySearchInput(BaseModel):
    query: str = Field(...,
                       description="The concept or entity to search for in the user's memory graph.")


class MemoryExtractionInput(BaseModel):
    entities: list = Field(
        ..., description="List of entity dictionaries with 'name', 'entityType', and 'observations' keys.")


def _cache_chart_data(payload_data):
    """Caches chart data into the local SQLite database to be retrieved instantly by the frontend."""
    cache_id = f"CACHE_{uuid.uuid4().hex}"
    with SessionLocal() as session:
        session.execute(text('''CREATE TABLE IF NOT EXISTS chart_cache (
                            cache_id TEXT PRIMARY KEY,
                            chart_data TEXT,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP)'''))
        session.execute(text('INSERT INTO chart_cache (cache_id, chart_data) VALUES (:cache_id, :chart_data)'),
                        {"cache_id": cache_id, "chart_data": json.dumps(payload_data)})
        session.commit()
    return cache_id


def _check_and_handle_sca_error(e, acc, user_uuid):
    """Checks if a TrueLayer API error is an SCA lock, and if so, returns an actionable authentication link."""
    if isinstance(e, PermissionError) and "SECURITY LOCK" in str(e):
        with SessionLocal() as session:
            row = session.execute(text("""
                SELECT b.truelayer_provider_id, b.refresh_token
                FROM banks b
                LEFT JOIN accounts a ON a.bank_uuid = b.bank_uuid
                WHERE (b.bank_name = :acc OR a.account_id = :acc) AND b.user_uuid = :user_uuid
            """), {"acc": acc, "user_uuid": user_uuid}).fetchone()

        token_gen = AccessTokenGenerator()
        auth_link = None

        if row and row[1]:
            provider_id = row[0]
            enc_key = os.getenv(
                "ENCRYPTION_KEY", b'cw_8H_1M4bX_3nF8vO5n3Y7A8xQ3_1m8aT2vP5_v5r8=')
            cipher_suite = Fernet(enc_key)
            refresh_token = cipher_suite.decrypt(row[1]).decode()

            auth_link = token_gen.get_reauth_link(refresh_token, user_uuid)

            if not auth_link:
                auth_link = token_gen.get_auth_link(user_uuid)
                auth_link += f"&provider_id={provider_id}"
        else:
            auth_link = token_gen.get_auth_link(user_uuid)

        return (f"⚠️ **SCA Security Lock Activated**\n\n"
                f"Under Open Banking regulations, {acc} requires you to periodically re-authenticate to access historical data older than 90 days.\n\n"
                f"**[Click here to securely re-authenticate with {acc}]({auth_link})**\n\n"
                f"Once you complete the bank prompt, come back and ask me to run this forecast again!")
    return None


def _parse_accounts(bank_name_or_id, user_uuid):
    """Resolves the requested bank names into actual identifiers available in the database."""
    if not bank_name_or_id or str(bank_name_or_id).lower() in ["none", ""]:
        return [], ""

    bank_name_or_id = str(bank_name_or_id).replace("'", "").replace("\’", "")

    if str(bank_name_or_id).upper() == "ALL":
        with SessionLocal() as session:
            accounts = [row[0] for row in session.execute(
                text("SELECT b.bank_name FROM banks b WHERE b.user_uuid = :user_uuid"), {"user_uuid": user_uuid}).fetchall()]
        return accounts, "ALL"
    else:
        with SessionLocal() as session:
            row = session.execute(text("""
                SELECT a.account_id, b.bank_name
                FROM accounts a
                JOIN banks b ON a.bank_uuid = b.bank_uuid
                WHERE (b.bank_name = :bank_name_or_id OR a.account_id = :bank_name_or_id) AND a.user_uuid = :user_uuid
            """), {"bank_name_or_id": bank_name_or_id, "user_uuid": user_uuid}).fetchone()

            if row:
                return [row[1]], row[0]

            return [bank_name_or_id], bank_name_or_id


def _get_combined_categorized_data(accounts, suffix, user_uuid):
    """Runs the categorization pipeline across all requested accounts and returns a merged dataframe."""
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
        except Exception as e:
            sca_msg = _check_and_handle_sca_error(e, acc, user_uuid)
            if sca_msg:
                raise Exception(sca_msg)
    return combined_df
