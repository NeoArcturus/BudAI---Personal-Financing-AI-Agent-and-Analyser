import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from sqlalchemy import text
from config import SessionLocal
from services.mcp_tools.shared_utils import GenerateExpenseForecastInput, _parse_accounts
from services.analytics.forecast_service import forecaster_engine
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

@tool(args_schema=GenerateExpenseForecastInput)
def generate_expense_forecast(
    user_uuid: str, 
    account_id: Optional[str] = None, 
    account_ids: Optional[List[str]] = None, 
    days: int = 30
) -> str:
    """
    Project future cumulative expenditure and compare current spending pace against
    prior month performance and active virtual bucket goals.
    """
    logger.info(json.dumps({"message": "Executing tool: generate_expense_forecast", "user_uuid": user_uuid, "days": days}))
    try:
        target_accounts = account_ids or ([account_id] if account_id else [])
        accounts, _ = _parse_accounts(target_accounts, user_uuid) if target_accounts else ([], "")
        
        target_account = None
        if accounts:
            target_account = accounts[0]
        else:
            with SessionLocal() as session:
                acc_row = session.execute(
                    text("SELECT account_id FROM accounts WHERE user_uuid = :u ORDER BY created_at ASC LIMIT 1"),
                    {"u": user_uuid}
                ).fetchone()
                if acc_row:
                    target_account = acc_row[0]

        agent = forecaster_engine
        current_balance = agent.fetch_live_balance(target_account, user_uuid) if target_account else 0.0
        E0, mu_E = agent.fetch_expense_parameters(target_account, user_uuid, 60) if target_account else (0.0, 0.0)
        df_temp, timeline = agent.run_expense_simulation(
            target_account or "primary", 
            E0, 
            mu_E, 
            user_uuid, 
            days=days, 
            paths=1000000, 
            current_balance=current_balance
        )

        projected_spend = 0.0
        if not df_temp.empty:
            exp_vals = df_temp.iloc[0].values.tolist()
            if exp_vals:
                projected_spend = round(float(exp_vals[min(days, len(exp_vals) - 1)]), 2)

        with SessionLocal() as session:
            # Query actual prior month spending (31 to 60 days ago) for run-rate comparison
            prior_spend_row = session.execute(
                text("""
                    SELECT COALESCE(SUM(ABS(amount)), 0.0)
                    FROM transactions
                    WHERE user_uuid = :u
                      AND is_pending = false
                      AND amount < 0
                      AND date >= CURRENT_DATE - INTERVAL '60 days'
                      AND date < CURRENT_DATE - INTERVAL '30 days'
                """),
                {"u": user_uuid}
            ).fetchone()
            prior_month_spend = round(float(prior_spend_row[0]) if prior_spend_row else 0.0, 2)

            # Query recent 30-day spending to date
            current_30d_spend_row = session.execute(
                text("""
                    SELECT COALESCE(SUM(ABS(amount)), 0.0)
                    FROM transactions
                    WHERE user_uuid = :u
                      AND is_pending = false
                      AND amount < 0
                      AND date >= CURRENT_DATE - INTERVAL '30 days'
                """),
                {"u": user_uuid}
            ).fetchone()
            current_30d_spend = round(float(current_30d_spend_row[0]) if current_30d_spend_row else 0.0, 2)

            # Query recent 30-day regular income / inflows
            income_row = session.execute(
                text("""
                    SELECT COALESCE(SUM(amount), 0.0)
                    FROM transactions
                    WHERE user_uuid = :u
                      AND is_pending = false
                      AND amount > 0
                      AND date >= CURRENT_DATE - INTERVAL '30 days'
                """),
                {"u": user_uuid}
            ).fetchone()
            monthly_income = round(float(income_row[0]) if income_row else 0.0, 2)

            # Query active goals from buckets table
            goals_query = session.execute(
                text("""
                    SELECT id, name, type, target_amount, target_date, cached_balance, priority_index
                    FROM buckets
                    WHERE user_id = :u
                      AND is_active = true
                      AND target_amount IS NOT NULL
                      AND target_amount > 0
                    ORDER BY priority_index ASC
                """),
                {"u": user_uuid}
            ).fetchall()

        now = datetime.now(timezone.utc)
        active_goals = []
        total_required_monthly_savings = 0.0

        for g in goals_query:
            g_id, g_name, g_type, g_target, g_date, g_cached, g_priority = g
            target_amt = float(g_target or 0.0)
            cached_amt = float(g_cached or 0.0)
            remaining_amt = max(0.0, target_amt - cached_amt)
            completion_pct = min(100.0, round((cached_amt / target_amt) * 100.0, 1)) if target_amt > 0 else 100.0

            # Calculate remaining months to target date
            if g_date:
                g_dt = g_date if g_date.tzinfo else g_date.replace(tzinfo=timezone.utc)
                days_left = max(1, (g_dt - now).days)
                months_left = max(1.0, days_left / 30.4375)
            else:
                months_left = 6.0  # default 6-month planning horizon

            monthly_allocation = round(remaining_amt / months_left, 2)
            total_required_monthly_savings += monthly_allocation

            active_goals.append({
                "name": g_name,
                "type": str(g_type),
                "target_amount": target_amt,
                "cached_balance": cached_amt,
                "remaining_amount": round(remaining_amt, 2),
                "completion_pct": completion_pct,
                "monthly_required_savings": monthly_allocation
            })

        # Evaluate goal presence and build output
        if active_goals:
            base_capacity = monthly_income if monthly_income > 0 else (prior_month_spend if prior_month_spend > 0 else current_balance)
            optimal_spend_ceiling = max(0.0, round(base_capacity - total_required_monthly_savings, 2))
            
            pacing_variance = round(projected_spend - optimal_spend_ceiling, 2)
            status_text = "ON_TRACK" if pacing_variance <= 0 else "OVERSPENDING_RISK"

            goals_summary_lines = []
            for goal in active_goals:
                goals_summary_lines.append(
                    f"  - {goal['name']} ({goal['type']}): Target £{goal['target_amount']:.2f}, "
                    f"Saved £{goal['cached_balance']:.2f} ({goal['completion_pct']}%), "
                    f"Monthly Allocation Needed: £{goal['monthly_required_savings']:.2f}"
                )
            goals_text = "\n".join(goals_summary_lines)

            output = (
                f"Expense Forecast ({days}-Day Horizon):\n"
                f"- Current Projected Spend: £{projected_spend:.2f}\n"
                f"- Prior Month Actual Spend: £{prior_month_spend:.2f}\n"
                f"- Optimal Spend Ceiling (Derived from Goals): £{optimal_spend_ceiling:.2f}\n"
                f"- Pacing Status: {status_text} (Variance: £{pacing_variance:+.2f})\n\n"
                f"Active Savings Goals:\n"
                f"{goals_text}\n"
                f"- Total Required Monthly Goal Savings: £{total_required_monthly_savings:.2f}"
            )
        else:
            spend_delta = round(projected_spend - prior_month_spend, 2)
            pct_delta = round((spend_delta / prior_month_spend) * 100.0, 1) if prior_month_spend > 0 else 0.0

            output = (
                f"Expense Forecast ({days}-Day Horizon):\n"
                f"- Current Projected Spend: £{projected_spend:.2f}\n"
                f"- Prior Month Actual Spend: £{prior_month_spend:.2f}\n"
                f"- Month-Over-Month Variance: £{spend_delta:+.2f} ({pct_delta:+.1f}%)\n\n"
                f"Note: No active goals found in virtual buckets. Optimal expenditure curve dropped; displaying monthly comparison only."
            )

        logger.info(json.dumps({"message": "generate_expense_forecast complete", "status_code": 200}))
        return output

    except Exception as e:
        logger.error(json.dumps({"message": f"Error in generate_expense_forecast: {e}", "status_code": 500}))
        return f"Error generating expense forecast: {str(e)}"
