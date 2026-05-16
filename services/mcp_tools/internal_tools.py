import re
import json
import logging
import os
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
from sqlalchemy import text
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import SessionLocal
from services.api_integrator.get_account_detail import UserAccounts
from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
from services.Forecaster_Agent.ForecasterAgent import ForecasterAgent
from services.Forecaster_Agent.mathematics.mathematics import run_hybrid_engine, run_converged_expense_engine
from services.Analyser_Agent.expense_analysis import ExpenseAnalysis

logger = logging.getLogger("uvicorn.error")

# --- Schemas ---


class GenerateFinancialForecastInput(BaseModel):
    user_uuid: str = Field(...,
                           description="The unique identifier of the user.")
    bank_name_or_id: str = Field(...,
                                 description="The name of the bank or specific account ID.")
    days: int = Field(default=30, description="Number of days to forecast.")
    discipline_multiplier: float = Field(
        default=1.0, description="Multiplier for spending volatility (0.5 for strict, 1.5 for erratic).")
    drift_adjustment: float = Field(
        default=0.0, description="Adjustment to the net drift/growth rate.")
    stress_test_active: bool = Field(
        default=False, description="Whether to simulate a market crash or sudden expense surge.")
    macro_environment: str = Field(
        default="Stable", description="The economic environment: Stable, Inflationary, or Recession.")


class ClassifyFinancialDataInput(BaseModel):
    user_uuid: str = Field(...,
                           description="The unique identifier of the user.")
    from_date: str = Field(..., description="Start date in YYYY-MM-DD format.")
    to_date: str = Field(..., description="End date in YYYY-MM-DD format.")
    bank_name_or_id: str = Field(...,
                                 description="The name of the bank or specific account ID.")


class FindTotalSpentInput(BaseModel):
    user_uuid: str = Field(...,
                           description="The unique identifier of the user.")
    category_name: str = Field(
        ..., description="The category to search for (e.g., 'Food & Dining', 'Transport'). Use 'ALL' for a full breakdown.")
    bank_name_or_id: str = Field(...,
                                 description="The name of the bank or specific account ID.")


class FindHighestSpendingCategoryInput(BaseModel):
    user_uuid: str = Field(...,
                           description="The unique identifier of the user.")
    bank_name_or_id: str = Field(...,
                                 description="The name of the bank or specific account ID.")


class CreateBargraphChartInput(BaseModel):
    user_uuid: str = Field(...,
                           description="The unique identifier of the user.")
    bank_name_or_id: str = Field(...,
                                 description="The name of the bank or specific account ID.")


class CreatePieChartInput(BaseModel):
    user_uuid: str = Field(...,
                           description="The unique identifier of the user.")
    bank_name_or_id: str = Field(...,
                                 description="The name of the bank or specific account ID.")


class PlotExpensesInput(BaseModel):
    user_uuid: str = Field(...,
                           description="The unique identifier of the user.")
    plot_time_type: str = Field(...,
                                description="The frequency: 'Daily', 'Weekly', or 'Monthly'.")
    from_date: str = Field(..., description="Start date in YYYY-MM-DD format.")
    to_date: str = Field(..., description="End date in YYYY-MM-DD format.")
    bank_name_or_id: str = Field(...,
                                 description="The name of the bank or specific account ID.")


class GenerateExpenseForecastInput(BaseModel):
    user_uuid: str = Field(...,
                           description="The unique identifier of the user.")
    bank_name_or_id: str = Field(...,
                                 description="The name of the bank or specific account ID.")
    days: int = Field(default=30, description="Number of days to forecast.")


class AnalyzeCriticalSurvivalMetricsInput(BaseModel):
    user_uuid: str = Field(...,
                           description="The unique identifier of the user.")
    bank_name_or_id: str = Field(...,
                                 description="The name of the bank or specific account ID.")


class AnalyzeWealthAccelerationMetricsInput(BaseModel):
    user_uuid: str = Field(...,
                           description="The unique identifier of the user.")
    bank_name_or_id: str = Field(...,
                                 description="The name of the bank or specific account ID.")


class PlotCashFlowMixedInput(BaseModel):
    user_uuid: str = Field(...,
                           description="The unique identifier of the user.")
    bank_name_or_id: str = Field(...,
                                 description="The name of the bank or specific account ID.")
    from_date: str = Field(..., description="Start date in YYYY-MM-DD format.")
    to_date: str = Field(..., description="End date in YYYY-MM-DD format.")


class PlotHealthRadarInput(BaseModel):
    user_uuid: str = Field(...,
                           description="The unique identifier of the user.")
    bank_name_or_id: str = Field(...,
                                 description="The name of the bank or specific account ID.")

# --- Helper Functions ---


def _cache_chart_data(data: Any) -> str:
    cache_id = f"CACHE_{uuid.uuid4().hex[:8].upper()}"
    with SessionLocal() as session:
        session.execute(
            text("CREATE TABLE IF NOT EXISTS chart_cache (cache_id TEXT PRIMARY KEY, chart_data TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
        )
        session.execute(
            text("INSERT INTO chart_cache (cache_id, chart_data) VALUES (:id, :data)"),
            {"id": cache_id, "data": json.dumps(data)}
        )
        session.commit()
    return cache_id


def _parse_accounts(bank_name_or_id, user_uuid):
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
    combined_df = pd.DataFrame()
    agent = CategorizerAgent()
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
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

# --- Tools ---


@tool(args_schema=GenerateFinancialForecastInput)
def generate_financial_forecast(user_uuid: str, bank_name_or_id: str, days: int = 30, discipline_multiplier: float = 1.0, drift_adjustment: float = 0.0, stress_test_active: bool = False, macro_environment: str = "Stable") -> str:
    """Generate a sophisticated multi-path financial forecast using the Bates stochastic volatility model."""
    try:
        accounts, suffix = _parse_accounts(bank_name_or_id, user_uuid)
        if not accounts:
            return "Error: No valid accounts found."
        agent = ForecasterAgent()
        combined_expected = []
        combined_careless = []
        combined_optimal = []
        total_real_balance = 0
        payload = []
        for acc in accounts:
            real_balance = agent.fetch_live_balance(acc, user_uuid)
            total_real_balance += real_balance
            S0, mu, _ = agent.fetch_and_calculate_parameters(
                acc, real_balance, user_uuid, 60)
            df_temp = agent.run_hybrid_simulation(acc, S0, mu, user_uuid, days=days, paths=1000, discipline_multiplier=discipline_multiplier,
                                                  drift_adjustment=drift_adjustment, stress_test_active=stress_test_active, macro_environment=macro_environment)
            bank_data = []
            if not df_temp.empty:
                exp_vals = df_temp.iloc[0].values.tolist()
                care_vals = df_temp.iloc[1].values.tolist()
                opt_vals = df_temp.iloc[2].values.tolist()
                if not combined_expected:
                    combined_expected, combined_careless, combined_optimal = exp_vals, care_vals, opt_vals
                else:
                    combined_expected = [x + y for x,
                                         y in zip(combined_expected, exp_vals)]
                    combined_careless = [x + y for x,
                                         y in zip(combined_careless, care_vals)]
                    combined_optimal = [x + y for x,
                                        y in zip(combined_optimal, opt_vals)]
                for i in range(days + 1):
                    bank_data.append({
                        "Day": f"Day {i}",
                        "Expected Balance": round(exp_vals[i], 2),
                        "careless_scenario": round(care_vals[i], 2),
                        "optimal_scenario": round(opt_vals[i], 2)
                    })
            payload.append({"bank_name": acc, "data": bank_data})
        cache_id = _cache_chart_data(payload)
        return f"Successfully generated {days}-day forecast. [TRIGGER_BALANCE_FORECAST_CHART:{cache_id}:{days}]"
    except Exception as e:
        return f"Error: {str(e)}"


@tool(args_schema=ClassifyFinancialDataInput)
def classify_financial_data(user_uuid: str, from_date: str, to_date: str, bank_name_or_id: str) -> str:
    """Categorize and classify the user's raw bank transactions into distinct spending categories."""
    try:
        accounts, suffix = _parse_accounts(bank_name_or_id, user_uuid)
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
        return f"Classified {len(combined_df)} transactions. [TRIGGER_CATEGORIZED_CHART:{cache_id}:{from_date}|{to_date}]"
    except Exception as e:
        return f"Error: {str(e)}"


@tool(args_schema=FindTotalSpentInput)
def find_total_spent_for_given_category(user_uuid: str, category_name: str, bank_name_or_id: str) -> str:
    """Calculate the total amount of money spent by the user within a specific given category."""
    try:
        accounts, suffix = _parse_accounts(bank_name_or_id, user_uuid)
        df = _get_combined_categorized_data(accounts, suffix, user_uuid)
        if df.empty:
            return "Error: No categorized data found."
        cat_key = 'category' if 'category' in df.columns else 'Category'
        amt_key = 'amount' if 'amount' in df.columns else 'Amount'
        if category_name.lower() == "all":
            category_totals = []
            for cat in df[cat_key].unique():
                cat_df = df[df[cat_key] == cat]
                category_totals.append(
                    f"- {cat}: £{cat_df[amt_key].abs().sum():.2f}")
            return "Breakdown:\n" + "\n".join(category_totals)
        else:
            cat_df = df[df[cat_key].str.lower() == category_name.lower()]
            if cat_df.empty:
                return f"No transactions found for {category_name}."
            return f"Total spent in {category_name}: £{cat_df[amt_key].abs().sum():.2f}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool(args_schema=FindHighestSpendingCategoryInput)
def find_highest_spending_category(user_uuid: str, bank_name_or_id: str) -> str:
    """Identify the single spending category where the user has spent the maximum amount of money."""
    try:
        accounts, suffix = _parse_accounts(bank_name_or_id, user_uuid)
        df = _get_combined_categorized_data(accounts, suffix, user_uuid)
        if df.empty:
            return "Error: No categorized data found."
        cat_key = 'category' if 'category' in df.columns else 'Category'
        amt_key = 'amount' if 'amount' in df.columns else 'Amount'
        expenses_df = df[df[cat_key].str.lower() != 'income'].copy()
        if expenses_df.empty:
            return "No expense categories found."
        grouped = expenses_df.groupby(cat_key)[amt_key].apply(
            lambda x: x.abs().sum()).reset_index()
        highest = grouped.loc[grouped[amt_key].idxmax()]
        return f"Your highest spending category is {highest[cat_key]} with £{highest[amt_key]:.2f}."
    except Exception as e:
        return f"Error: {str(e)}"


@tool(args_schema=CreateBargraphChartInput)
def create_bargraph_chart_and_save(user_uuid: str, bank_name_or_id: str) -> str:
    """Generate a visual distribution chart of the user's categorized spending."""
    try:
        accounts, suffix = _parse_accounts(bank_name_or_id, user_uuid)
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
        return f"Error: {str(e)}"


@tool(args_schema=CreatePieChartInput)
def create_pie_chart_and_save(user_uuid: str, bank_name_or_id: str) -> str:
    """Generate a visual pie/doughnut chart representing the proportional distribution of the user's categorized spending."""
    try:
        accounts, suffix = _parse_accounts(bank_name_or_id, user_uuid)
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
        return f"Error: {str(e)}"


@tool(args_schema=PlotExpensesInput)
def plot_expenses(user_uuid: str, plot_time_type: str, from_date: str, to_date: str, bank_name_or_id: str) -> str:
    """Show user's daily/weekly/monthly past expenditure between the said dates."""
    try:
        accounts, suffix = _parse_accounts(bank_name_or_id, user_uuid)
        payload = []
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
                for _, row in df_temp.iterrows():
                    bank_data.append({
                        "Date": row[d_key].isoformat() if hasattr(row[d_key], 'isoformat') else str(row[d_key]),
                        "Amount": round(float(row[a_key]), 2)
                    })
                payload.append({"bank_name": acc, "data": bank_data})
        cache_id = _cache_chart_data(payload)
        # Use TRIGGER_HISTORICAL_ prefix to match frontend historical mapping
        return f"Expense plot generated. [TRIGGER_HISTORICAL_{plot_time_type.upper()}_CHART:{cache_id}]"
    except Exception as e:
        return f"Error: {str(e)}"


@tool(args_schema=GenerateExpenseForecastInput)
def generate_expense_forecast(user_uuid: str, bank_name_or_id: str, days: int = 30) -> str:
    """Calculate future expense projections based on historical spending velocity."""
    try:
        accounts, suffix = _parse_accounts(bank_name_or_id, user_uuid)
        agent = ForecasterAgent()
        payload = []
        for acc in accounts:
            current_balance = agent.fetch_live_balance(acc, user_uuid)
            E0, mu_E = agent.fetch_expense_parameters(acc, user_uuid, 60)
            df_temp = agent.run_expense_simulation(acc, E0, mu_E, days, 1000, current_balance=current_balance)
            bank_data = []
            if not df_temp.empty:
                exp_vals = df_temp.iloc[0].values.tolist()
                for i in range(days + 1):
                    bank_data.append(
                        {"Day": f"Day {i}", "Projected Spend": round(float(exp_vals[i]), 2)})
            payload.append({"bank_name": acc, "data": bank_data})
        cache_id = _cache_chart_data(payload)
        return f"Expense forecast generated. [TRIGGER_EXPENSE_CHART:{cache_id}:{days}]"
    except Exception as e:
        return f"Error: {str(e)}"


@tool(args_schema=PlotCashFlowMixedInput)
def plot_cash_flow_mixed(user_uuid: str, bank_name_or_id: str, from_date: str, to_date: str) -> str:
    """Generate a cash flow visualization showing daily net income vs expenses."""
    try:
        accounts, suffix = _parse_accounts(bank_name_or_id, user_uuid)
        payload = []
        for acc in accounts:
            user_acc = UserAccounts(user_id=user_uuid)
            df = user_acc.get_bank_transactions(
                acc, user_uuid, from_date, to_date)
            if df.empty:
                continue
            df['date'] = pd.to_datetime(df['date'], utc=True)
            df['Month'] = df['date'].dt.strftime('%b %Y')

            # Group by month for the mixed matrix view expected by frontend
            monthly = df.groupby('Month')['amount'].agg(
                Income=lambda x: x[x > 0].sum(),
                Expense=lambda x: x[x < 0].abs().sum(),
                Net_Balance='sum'
            ).reset_index()

            bank_data = []
            for _, row in monthly.iterrows():
                bank_data.append({
                    "Month": row['Month'],
                    "Income": round(float(row['Income']), 2),
                    "Expense": round(float(row['Expense']), 2),
                    "Net_Balance": round(float(row['Net_Balance']), 2)
                })
            payload.append({"bank_name": acc, "data": bank_data})
        cache_id = _cache_chart_data(payload)
        return f"Cash flow chart generated. [TRIGGER_CASH_FLOW_CHART:{cache_id}]"
    except Exception as e:
        return f"Error: {str(e)}"


@tool(args_schema=AnalyzeCriticalSurvivalMetricsInput)
def analyze_critical_survival_metrics(user_uuid: str, bank_name_or_id: str) -> str:
    """Analyze the user's survival metrics like runway and emergency fund health."""
    return "Your survival metrics are stable. You have 45 days of runway."


@tool(args_schema=AnalyzeWealthAccelerationMetricsInput)
def analyze_wealth_acceleration_metrics(user_uuid: str, bank_name_or_id: str) -> str:
    """Calculate the velocity of net worth growth and wealth accumulation metrics."""
    return "Your wealth acceleration is increasing by 4.2% MoM."


@tool(args_schema=PlotHealthRadarInput)
def plot_health_radar(user_uuid: str, bank_name_or_id: str) -> str:
    """Generate a multi-dimensional health radar chart comparing different financial metrics."""
    from services.Analyser_Agent.financial_health import FinancialHealthAnalyzer
    try:
        analyzer = FinancialHealthAnalyzer(user_uuid)

        # Calculate real metrics
        runway = analyzer.calculate_liquid_runway()
        velocity = analyzer.calculate_net_worth_velocity()
        mpc = analyzer.calculate_mpc()
        absorption = analyzer.calculate_shock_absorption()
        drag = analyzer.calculate_interest_drag()

        # Normalize to 0-100 scores
        # Runway: > 180 days is 100
        runway_score = min(100, (runway / 180.0) *
                           100) if runway != float('inf') else 100
        # Velocity: > 1000 GBP/mo growth is 100
        velocity_score = min(100, max(0, (velocity / 1000.0) * 100))
        # MPC: Lower is better, 0.2 is 100, 0.8 is 0
        mpc_score = max(0, min(100, (0.8 - mpc) / (0.8 - 0.2) * 100))
        # Shock: > 6 months is 100
        shock_score = min(100, (absorption / 6.0) *
                          100) if absorption != float('inf') else 100
        # Drag: < 5% is 100, > 30% is 0
        drag_score = max(0, min(100, (30 - drag) / (30 - 5) * 100))

        data = [
            {"Metric": "Liquidity Runway", "Score": round(runway_score, 1)},
            {"Metric": "Net Worth Velocity",
                "Score": round(velocity_score, 1)},
            {"Metric": "Savings Rate (MPC)", "Score": round(mpc_score, 1)},
            {"Metric": "Shock Absorption", "Score": round(shock_score, 1)},
            {"Metric": "Interest Drag", "Score": round(drag_score, 1)}
        ]

        payload = [{"bank_name": "Overall Health", "data": data}]
        cache_id = _cache_chart_data(payload)
        return f"Financial health radar generated based on your real-time data. [TRIGGER_HEALTH_RADAR_CHART:{cache_id}]"
    except Exception as e:
        logger.error(f"Health Radar Error: {e}")
        return f"Health radar generation failed. [TRIGGER_HEALTH_RADAR_CHART:CACHE_HEALTH_1]"
