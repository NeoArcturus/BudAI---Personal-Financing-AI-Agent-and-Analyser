import logging
from langchain_core.tools import tool
from services.mcp_tools.shared_utils import (
    GenerateFinancialForecastInput, GenerateExpenseForecastInput,
    GenerateHypotheticalScenarioInput, ForecastBudgetImpactInput,
    _cache_chart_data, _parse_accounts
)
from services.Forecaster_Agent.ForecasterAgent import ForecasterAgent
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)


@tool(args_schema=GenerateHypotheticalScenarioInput)
def generate_hypothetical_scenario(user_uuid: str, account_ids: list[str], days: int = 30, injections: list = None) -> str:
    """
    Generate a financial forecast based on a hypothetical scenario with multiple custom financial injections.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        account_ids (list[str]): The target account (must be exactly one).
        days (int): The number of days to forecast.
        injections (list, optional): List of custom injections/events to apply.
        
    Returns:
        str: The scenario outcome and chart trigger payload.
    """
    logger.info(f"Executing MCP Tool: generate_hypothetical_scenario")
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

        _res = f"Scenario simulation complete. Final projected balance at day {days}:{summary_text}\n\nINJECTED SCENARIO EVENTS:\n{events_str}\n\n[TRIGGER_BALANCE_FORECAST_CHART:{cache_id}]"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res
    except Exception as e:
        logger.error(f"Scenario error: {e}")
        _res = f"Error: {str(e)}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res

@tool(args_schema=GenerateFinancialForecastInput)
def generate_financial_forecast(user_uuid: str, account_ids: list[str], days: int = 30, discipline_multiplier: float = 1.0, drift_adjustment: float = 0.0, stress_test_active: bool = False, macro_environment: str = "Stable") -> str:
    """
    Generate a high-precision multi-path financial forecast with 1 million paths and deterministic transaction mapping.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        account_ids (list[str]): The target account (must be exactly one).
        days (int): The number of days to forecast.
        discipline_multiplier (float): Factor to scale expenditure discipline.
        drift_adjustment (float): Factor to adjust the geometric drift.
        stress_test_active (bool): Whether to run adverse stress scenarios.
        macro_environment (str): The macroeconomic assumption (e.g. 'Stable').
        
    Returns:
        str: The detailed financial forecast and chart trigger payload.
    """
    logger.info(f"Executing MCP Tool: generate_financial_forecast")
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

        _res = f"Forecast generated with 1,000,000 paths. Final projections at day {days}:{summary_text}\n\nPROJECTED EVENTS:\n{events_str}\n\n[TRIGGER_BALANCE_FORECAST_CHART:{cache_id}]"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        _res = f"Error: {str(e)}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res

@tool(args_schema=GenerateExpenseForecastInput)
def generate_expense_forecast(user_uuid: str, account_ids: list[str], days: int = 30) -> str:
    """
    Calculate future expense projections using 1 million paths and historical spending velocity.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        account_ids (list[str]): The target account (must be exactly one).
        days (int): The number of days to forecast.
        
    Returns:
        str: The projected cumulative spend at the end of the forecast period and chart trigger.
    """
    logger.info(f"Executing MCP Tool: generate_expense_forecast")
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
        last = None
        for p in payload:
            if p["data"]:
                last = p["data"][-1]
                summary_text += f"\n- Account {p['bank_name']} (Day {days}): Projected Cumulative Spend £{last['Projected Spend']}"

        if last:
            _res = f"Expense forecast generated. Final expected cumulative spend at day {days}: £{last['Projected Spend']}\n\n[TRIGGER_EXPENSE_FORECAST_CHART:{cache_id}]"
        else:
            _res = f"Expense forecast generated, but no projection data is available.\n\n[TRIGGER_EXPENSE_FORECAST_CHART:{cache_id}]"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res
    except Exception as e:
        logger.error(f"Error generating expense forecast: {e}")
        _res = f"Error: {str(e)}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res

@tool(args_schema=ForecastBudgetImpactInput)
def forecast_budget_impact(user_uuid: str, account_ids: list[str], days: int = 30) -> str:
    """
    Run a detailed financial forecast incorporating the user's live budget pacing (Spend Velocity & Variance) to clamp the deterministic floor in the Monte Carlo simulation.
    
    Args:
        user_uuid (str): The unique identifier of the user.
        account_ids (list[str]): The target account (must be exactly one).
        days (int): The number of days to forecast.
        
    Returns:
        str: The impact analysis and simulated forecast outcomes.
    """
    logger.info(f"Executing MCP Tool: forecast_budget_impact")
    try:
        from services.Analyser_Agent.budget_engine import BudgetEngine
        engine = BudgetEngine(user_uuid)
        budgets = engine.get_variance_for_category(None)
        
        total_overspend = 0.0
        for b in budgets:
            if b['variance'] < 0:
                total_overspend += abs(b['variance'])
                
        injections = []
        if total_overspend > 0:
            injections.append({
                "day": min(28, days),
                "amount": -total_overspend,
                "type": "one-off",
                "description": f"Projected Budget Overspend across {len([b for b in budgets if b['variance'] < 0])} categories"
            })
            
        _res = generate_hypothetical_scenario.invoke({
            "user_uuid": user_uuid,
            "account_ids": account_ids,
            "days": days,
            "injections": injections
        })
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res
    except Exception as e:
        logger.error(f"Error forecasting budget impact: {e}")
        _res = f"Error: {str(e)}"
        logger.info(f"Tool returned: {str(_res)[:1000]}")
        return _res
