import logging
from langchain_core.tools import tool
from services.mcp_tools.shared_utils import (
    GenerateFinancialForecastInput, GenerateExpenseForecastInput,
    _cache_chart_data, _parse_accounts
)
from services.Forecaster_Agent.ForecasterAgent import ForecasterAgent
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

@tool(args_schema=GenerateFinancialForecastInput)
def generate_financial_forecast(user_uuid: str, bank_name_or_id: str, days: int = 30, discipline_multiplier: float = 1.0, drift_adjustment: float = 0.0, stress_test_active: bool = False, macro_environment: str = "Stable") -> str:
    """Generate a sophisticated multi-path financial forecast using the Bates stochastic volatility model."""
    logger.info(f"Generating financial forecast for user: {user_uuid}, days: {days}")
    try:
        accounts, suffix = _parse_accounts(bank_name_or_id, user_uuid)
        if not accounts:
            logger.warning("No valid accounts found for forecast.")
            return "Error: No valid accounts found."
        agent = ForecasterAgent()
        combined_expected = []
        combined_careless = []
        combined_optimal = []
        payload = []
        for acc in accounts:
            logger.debug(f"Processing forecast for account: {acc}")
            real_balance = agent.fetch_live_balance(acc, user_uuid)
            S0, mu, _ = agent.fetch_and_calculate_parameters(
                acc, real_balance, user_uuid, 60)
            df_temp = agent.run_hybrid_simulation(acc, S0, mu, user_uuid, days=days, paths=1000, discipline_multiplier=discipline_multiplier,
                                                  drift_adjustment=drift_adjustment, stress_test_active=stress_test_active, macro_environment=macro_environment)
            bank_data = []
            if not df_temp.empty:
                exp_vals = df_temp.iloc[0].values.tolist()
                care_vals = df_temp.iloc[1].values.tolist()
                opt_vals = df_temp.iloc[2].values.tolist()
                for i in range(days + 1):
                    bank_data.append({
                        "Day": f"Day {i}",
                        "Expected Balance": round(exp_vals[i], 2),
                        "careless_scenario": round(care_vals[i], 2),
                        "optimal_scenario": round(opt_vals[i], 2)
                    })
            payload.append({"bank_name": acc, "data": bank_data})
        cache_id = _cache_chart_data(payload)
        logger.info(f"Forecast completed for user: {user_uuid}")
        return f"Successfully generated {days}-day forecast. [TRIGGER_BALANCE_FORECAST_CHART:{cache_id}:{days}]"
    except Exception as e:
        logger.error(f"Forecast generation failed: {e}", exc_info=True)
        return f"Error: {str(e)}"

@tool(args_schema=GenerateExpenseForecastInput)
def generate_expense_forecast(user_uuid: str, bank_name_or_id: str, days: int = 30) -> str:
    """Calculate future expense projections based on historical spending velocity."""
    logger.info(f"Generating expense forecast for user: {user_uuid}, days: {days}")
    try:
        accounts, suffix = _parse_accounts(bank_name_or_id, user_uuid)
        from services.Forecaster_Agent.ForecasterAgent import ForecasterAgent
        agent = ForecasterAgent()
        payload = []
        for acc in accounts:
            logger.debug(f"Forecasting expenses for account: {acc}")
            current_balance = agent.fetch_live_balance(acc, user_uuid)
            E0, mu_E = agent.fetch_expense_parameters(acc, user_uuid, 60)
            df_temp = agent.run_expense_simulation(
                acc, E0, mu_E, days, 1000, current_balance=current_balance)
            bank_data = []
            if not df_temp.empty:
                exp_vals = df_temp.iloc[0].values.tolist()
                for i in range(days + 1):
                    bank_data.append(
                        {"Day": f"Day {i}", "Projected Spend": round(float(exp_vals[i]), 2)})
            payload.append({"bank_name": acc, "data": bank_data})
        cache_id = _cache_chart_data(payload)
        logger.info(f"Expense forecast generated with cache ID: {cache_id}")
        return f"Expense forecast generated. [TRIGGER_EXPENSE_CHART:{cache_id}:{days}]"
    except Exception as e:
        logger.error(f"Failed to generate expense forecast: {e}", exc_info=True)
        return f"Error: {str(e)}"
