import logging
from langchain_core.tools import tool
from services.mcp_tools.shared_utils import (
    GenerateFinancialForecastInput, GenerateExpenseForecastInput,
    GenerateHypotheticalScenarioInput,
    _cache_chart_data, _parse_accounts
)
from services.Forecaster_Agent.ForecasterAgent import ForecasterAgent
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

@tool(args_schema=GenerateHypotheticalScenarioInput)
def generate_hypothetical_scenario(user_uuid: str, account_ids: list[str], days: int = 30, injections: list = None) -> str:
    """Generate a financial forecast based on a hypothetical scenario with multiple custom financial injections."""
    try:
        if injections is None: injections = []
        accounts, suffix = _parse_accounts(account_ids, user_uuid)
        if not accounts: return "Error: No accounts found."
        if len(accounts) > 1: return "Please specify exactly one account for scenario analysis."
        
        agent = ForecasterAgent()
        payload, timeline_events = [], []
        
        clean_injections = []
        for inj in injections:
            if hasattr(inj, 'dict'):
                clean_injections.append(inj.dict())
            else:
                clean_injections.append(inj)

        for acc in accounts:
            real_balance = agent.fetch_live_balance(acc, user_uuid)
            S0, mu, _ = agent.fetch_and_calculate_parameters(acc, real_balance, user_uuid, 60)
            
            df_temp, timeline = agent.run_scenario_simulation(
                acc, S0, mu, user_uuid, days=days, paths=1000000, 
                injections=clean_injections
            )
            
            bank_data = []
            if not df_temp.empty:
                exp_vals, care_vals, opt_vals = df_temp.iloc[0].values.tolist(), df_temp.iloc[1].values.tolist(), df_temp.iloc[2].values.tolist()
                for i in range(days + 1):
                    bank_data.append({
                        "Day": f"Day {i}", 
                        "Expected Balance": round(exp_vals[i], 2), 
                        "careless_scenario": round(care_vals[i], 2), 
                        "optimal_scenario": round(opt_vals[i], 2)
                    })
            payload.append({"bank_name": acc, "data": bank_data})
            timeline_events.extend(timeline)
        
        cache_id = _cache_chart_data({"series": payload, "timeline": timeline_events})
        
        scenario_events = [e for e in timeline_events if "[SCENARIO]" in e.get('merchant', '')]
        events_str = "\n".join([f"- Day {e['day']}: {e['merchant']} £{e['amount']}" for e in scenario_events])
        
        summary_text = ""
        for p in payload:
            if p["data"]:
                last = p["data"][-1]
                summary_text += f"\n- Account {p['bank_name']} (Day {days}): Projected Balance £{last['Expected Balance']}"

        return f"Scenario simulation complete. Final projected balance at day {days}:{summary_text}\n\nINJECTED SCENARIO EVENTS:\n{events_str}\n\n[TRIGGER_BALANCE_FORECAST_CHART:{cache_id}]"
    except Exception as e:
        logger.error(f"Scenario error: {e}")
        return f"Error: {str(e)}"

@tool(args_schema=GenerateFinancialForecastInput)
def generate_financial_forecast(user_uuid: str, account_ids: list[str], days: int = 30, discipline_multiplier: float = 1.0, drift_adjustment: float = 0.0, stress_test_active: bool = False, macro_environment: str = "Stable") -> str:
    """Generate a high-precision multi-path financial forecast with 1 million paths and deterministic transaction mapping."""
    try:
        accounts, suffix = _parse_accounts(account_ids, user_uuid)
        if not accounts: return "Error: No accounts found."
        if len(accounts) > 1: return "Please specify exactly one account for forecasting. The ForecasterAgent cannot process multiple accounts simultaneously."
        
        agent = ForecasterAgent()
        payload, timeline_events = [], []
        for acc in accounts:
            real_balance = agent.fetch_live_balance(acc, user_uuid)
            S0, mu, _ = agent.fetch_and_calculate_parameters(acc, real_balance, user_uuid, 60)
            df_temp, timeline = agent.run_hybrid_simulation(acc, S0, mu, user_uuid, days=days, paths=1000000, discipline_multiplier=discipline_multiplier, drift_adjustment=drift_adjustment, stress_test_active=stress_test_active, macro_environment=macro_environment)
            bank_data = []
            if not df_temp.empty:
                exp_vals, care_vals, opt_vals = df_temp.iloc[0].values.tolist(), df_temp.iloc[1].values.tolist(), df_temp.iloc[2].values.tolist()
                for i in range(days + 1):
                    bank_data.append({"Day": f"Day {i}", "Expected Balance": round(exp_vals[i], 2), "careless_scenario": round(care_vals[i], 2), "optimal_scenario": round(opt_vals[i], 2)})
            payload.append({"bank_name": acc, "data": bank_data})
            timeline_events.extend(timeline)
        
        cache_id = _cache_chart_data({"series": payload, "timeline": timeline_events})
        events_str = "\n".join([f"- Day {e['day']}: {e['merchant']} ({e['category']}) £{e['amount']}" for e in timeline_events])
        
        summary_text = ""
        for p in payload:
            if p["data"]:
                last = p["data"][-1]
                summary_text += f"\n- Account {p['bank_name']} (Day {days}): Expected £{last['Expected Balance']}, Careless £{last['careless_scenario']}, Optimal £{last['optimal_scenario']}"

        return f"Forecast generated with 1,000,000 paths. Final projections at day {days}:{summary_text}\n\nPROJECTED EVENTS:\n{events_str}\n\n[TRIGGER_BALANCE_FORECAST_CHART:{cache_id}]"
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return f"Error: {str(e)}"

@tool(args_schema=GenerateExpenseForecastInput)
def generate_expense_forecast(user_uuid: str, account_ids: list[str], days: int = 30) -> str:
    """Calculate future expense projections using 1 million paths and historical spending velocity."""
    try:
        accounts, suffix = _parse_accounts(account_ids, user_uuid)
        if not accounts: return "Error: No accounts found."
        if len(accounts) > 1: return "Please specify exactly one account for forecasting. The ForecasterAgent cannot process multiple accounts simultaneously."
        
        agent = ForecasterAgent()
        payload, timeline_events = [], []
        for acc in accounts:
            current_balance = agent.fetch_live_balance(acc, user_uuid)
            E0, mu_E = agent.fetch_expense_parameters(acc, user_uuid, 60)
            df_temp, timeline = agent.run_expense_simulation(acc, E0, mu_E, user_uuid, days, 1000000, current_balance=current_balance)
            bank_data = []
            if not df_temp.empty:
                exp_vals = df_temp.iloc[0].values.tolist()
                for i in range(days + 1):
                    bank_data.append({"Day": f"Day {i}", "Projected Spend": round(float(exp_vals[i]), 2)})
            payload.append({"bank_name": acc, "data": bank_data})
            timeline_events.extend(timeline)
        
        cache_id = _cache_chart_data({"series": payload, "timeline": timeline_events})
        events_str = "\n".join([f"- Day {e['day']}: {e['merchant']} ({e['category']}) £{e['amount']}" for e in timeline_events])
        
        summary_text = ""
        for p in payload:
            if p["data"]:
                last = p["data"][-1]
                summary_text += f"\n- Account {p['bank_name']} (Day {days}): Projected Cumulative Spend £{last['Projected Spend']}"

        return f"Expense forecast generated with 1,000,000 paths. Final projection at day {days}:{summary_text}\n\nPROJECTED EVENTS:\n{events_str}\n\n[TRIGGER_EXPENSE_CHART:{cache_id}]"
    except Exception as e:
        logger.error(f"Expense error: {e}")
        return f"Error: {str(e)}"
