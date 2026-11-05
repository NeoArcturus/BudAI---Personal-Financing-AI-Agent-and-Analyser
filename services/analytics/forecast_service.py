import json
import os
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from sqlalchemy import text
from config import SessionLocal
from services.api_integrator.account_reader import AccountReader
from agents.core_financial.Forecaster_Agent.mathematics.mathematics import (
    run_hybrid_engine,
    run_converged_expense_engine
)
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)


class ForecasterEngine:
    """
    Mathematical Forecaster Engine executing continuous-time Heston/Bates 
    stochastic simulations and empirical spending velocity projections.
    """
    def __init__(self):
        pass

    def fetch_live_balance(self, identifier: str, user_uuid: str) -> float:
        reader = AccountReader(user_id=user_uuid)
        balance = reader.get_account_balance(identifier, user_uuid)
        return float(balance) if balance is not None else 0.0

    def get_user_params(self, user_uuid: str) -> Dict[str, float]:
        return {
            'kappa': 0.1,
            'theta': 1000.0,
            'xi': 0.2,
            'rho': -0.5,
            'lambda': 0.1,
            'mu_J': -0.05,
            'sigma_j': 0.1
        }

    def fetch_and_calculate_parameters(
        self, account_id: str, current_balance: float, user_uuid: str, lookback_days: int = 60
    ) -> Tuple[float, float, float]:
        try:
            reader = AccountReader(user_uuid)
            df = reader.get_transactions(account_id, user_uuid=user_uuid)
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            return current_balance, -0.01, 0.05

        df['Date'] = pd.to_datetime(
            df.get('date', df.get('timestamp')), 
            format='ISO8601', 
            utc=True
        ).dt.date
        
        daily_net = df.groupby('Date')['amount'].sum().reset_index().sort_values('Date')
        daily_net['Reverse_Amount'] = daily_net['amount'].iloc[::-1]
        historical_balances, temp_balance = [current_balance], current_balance
        for amt in daily_net['Reverse_Amount'].values[:-1]:
            temp_balance -= amt
            historical_balances.append(temp_balance)
        daily_net['Balance'] = historical_balances[::-1]
        
        recent_data = daily_net.tail(lookback_days).copy()
        recent_data['Safe_Balance'] = recent_data['Balance'].apply(lambda x: max(x, 10))
        recent_data['Returns'] = np.log(
            recent_data['Safe_Balance'] / recent_data['Safe_Balance'].shift(1)
        )
        mu = float(np.clip(recent_data['Returns'].mean(), -0.02, 0.02))
        sigma = recent_data['Returns'].std()
        return (
            current_balance, 
            (mu if not np.isnan(mu) else -0.01), 
            (sigma if not np.isnan(sigma) and sigma != 0 else 0.05)
        )

    def fetch_expense_parameters(
        self, account_id: str, user_uuid: str, lookback_days: int = 60
    ) -> Tuple[float, float]:
        try:
            reader = AccountReader(user_uuid)
            df = reader.get_transactions(account_id, user_uuid=user_uuid)
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            return 50.0, 0.001

        df['Date'] = pd.to_datetime(
            df.get('date', df.get('timestamp')), 
            format='ISO8601', 
            utc=True
        ).dt.date
        df['amount'] = df['amount'].abs()
        recent_data = df.groupby('Date')['amount'].sum().reset_index().sort_values('Date').tail(lookback_days).copy()
        recent_data['Safe_Amount'] = recent_data['amount'].apply(lambda x: max(x, 1))
        E0 = recent_data['Safe_Amount'].mean()
        recent_data['Returns'] = np.log(
            recent_data['Safe_Amount'] / recent_data['Safe_Amount'].shift(1)
        )
        mu_E = float(np.clip(recent_data['Returns'].mean(), -0.02, 0.02))
        return float(E0), (mu_E if not np.isnan(mu_E) else 0.001)

    def detect_upcoming_expenses(
        self, user_uuid: str, account_id: Optional[str] = None, days: int = 30, df_input: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if df_input is not None:
            df = df_input.copy()
        else:
            reader = AccountReader(user_id=user_uuid)
            df = reader.get_transactions(account_id, user_uuid)

        if df.empty:
            return np.zeros(days + 1), []

        df['Date'] = pd.to_datetime(
            df.get('date', df.get('timestamp')), format='ISO8601', utc=True
        )
        critical_categories = [
            "Bills & Utilities", "Housing", "Entertainment", 
            "Transfers & Investments", "Taxes", "Groceries", "Dining Out"
        ]
        df_out = df[(df['amount'] < 0) & (df['category'].isin(critical_categories))].copy()

        calendar = np.zeros(days + 1)
        timeline = []
        now = pd.Timestamp.utcnow()

        df_out['group_desc'] = df_out['description'].str.replace(r'\d+\.\d+\s?\w*', '', regex=True).str.strip()

        for (desc_key, cat), group in df_out.groupby(['group_desc', 'category']):
            if len(group) >= 2:
                group = group.sort_values('Date')
                deltas = group['Date'].diff().dt.days.dropna()
                median_delta = deltas.median()
                max_std = 5 if len(group) <= 5 else 8

                if 25 <= median_delta <= 35 and deltas.std() < max_std:
                    projected_amount = group['amount'].iloc[-1]
                    display_desc = group['description'].iloc[-1]
                    last_date = group['Date'].iloc[-1]
                    next_date = last_date + pd.Timedelta(days=round(median_delta))
                    while (next_date - now).days <= days:
                        day_idx = (next_date - now).days
                        if 0 <= day_idx <= days:
                            calendar[day_idx] += projected_amount
                            timeline.append({
                                'day': day_idx,
                                'date': next_date.strftime('%Y-%m-%d'),
                                'merchant': display_desc,
                                'category': cat,
                                'amount': round(float(projected_amount), 2)
                            })
                        next_date += pd.Timedelta(days=round(median_delta))

        return calendar, sorted(timeline, key=lambda x: x['day'])

    def run_hybrid_simulation(
        self, account_id: str, S0: float, mu: float, user_uuid: str,
        days: int = 60, paths: int = 1000000, discipline_multiplier: float = 1.0,
        drift_adjustment: float = 0.0, stress_test_active: bool = False,
        macro_environment: str = "Stable"
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        params = self.get_user_params(user_uuid)
        if discipline_multiplier != 1.0 or drift_adjustment != 0.0 or stress_test_active or macro_environment != "Stable":
            if discipline_multiplier < 1.0:
                params['theta'] *= 0.8
                params['xi'] *= 0.7
            elif discipline_multiplier > 1.0:
                params['theta'] *= 1.4
                params['xi'] *= 1.5
            mu += drift_adjustment
            if macro_environment == "Inflationary":
                params['theta'] *= 1.2
                mu -= 0.005
            elif macro_environment == "Recession":
                params['lambda'] += 0.2
                params['mu_J'] -= 0.1
            if stress_test_active:
                params['lambda'], params['mu_J'], params['sigma_j'] = 0.8, -0.4, 0.2

        calendar, timeline = self.detect_upcoming_expenses(user_uuid, account_id, days)
        df = run_hybrid_engine(S0, mu, params, days, paths, account_id, calendar)
        return df, timeline

    def run_expense_simulation(
        self, account_id: str, E0: float, mu: float, user_uuid: str,
        days: int = 30, paths: int = 1000000, discipline_multiplier: float = 1.0,
        drift_adjustment: float = 0.0, stress_test_active: bool = False,
        macro_environment: str = "Stable", current_balance: Optional[float] = None
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        calendar, timeline = self.detect_upcoming_expenses(user_uuid, account_id, days)
        if np.all(calendar == 0):
            df = pd.DataFrame(np.zeros((1, days + 1)))
            return df, timeline

        if discipline_multiplier != 1.0 or drift_adjustment != 0.0 or stress_test_active or macro_environment != "Stable":
            if discipline_multiplier < 1.0:
                mu -= 0.01
            elif discipline_multiplier > 1.0:
                mu += 0.02
            mu += drift_adjustment
            if macro_environment == "Inflationary":
                mu += 0.015
            elif macro_environment == "Recession":
                mu -= 0.01

        df = run_converged_expense_engine(E0, mu, days, paths, account_id, calendar)
        if current_balance is not None and not df.empty:
            for row_idx in range(len(df)):
                path_vals = df.iloc[row_idx].values.copy()
                path_vals = np.minimum(path_vals, current_balance)
                df.iloc[row_idx] = path_vals
        return df, timeline

    def run_scenario_simulation(
        self, account_id: str, S0: float, mu: float, user_uuid: str,
        days: int = 60, paths: int = 1000000, injections: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        if injections is None:
            injections = []
        params = self.get_user_params(user_uuid)
        calendar, timeline = self.detect_upcoming_expenses(user_uuid, account_id, days)

        for inj in injections:
            day = inj.get('day', 0)
            amount = inj.get('amount', 0.0)
            inj_type = inj.get('type', 'one-off')
            desc = inj.get('description', 'Scenario Event')

            if inj_type == 'one-off':
                if 0 <= day <= days:
                    calendar[day] += amount
                    timeline.append({
                        'day': day,
                        'date': (pd.Timestamp.utcnow() + pd.Timedelta(days=day)).strftime('%Y-%m-%d'),
                        'merchant': f"[SCENARIO] {desc}",
                        'category': 'Scenario Injection',
                        'amount': round(float(amount), 2)
                    })
            elif inj_type == 'recurring_monthly':
                curr_day = day
                while curr_day <= days:
                    calendar[curr_day] += amount
                    timeline.append({
                        'day': curr_day,
                        'date': (pd.Timestamp.utcnow() + pd.Timedelta(days=curr_day)).strftime('%Y-%m-%d'),
                        'merchant': f"[SCENARIO] {desc}",
                        'category': 'Scenario Injection',
                        'amount': round(float(amount), 2)
                    })
                    curr_day += 30

        df = run_hybrid_engine(S0, mu, params, days, paths, account_id, calendar)
        return df, sorted(timeline, key=lambda x: x['day'])


forecaster_engine = ForecasterEngine()


def compute_balance_forecast(
    user_uuid: str,
    account_id: str,
    days: int = 30,
    discipline_multiplier: float = 1.0,
    drift_adjustment: float = 0.0,
    stress_test_active: bool = False,
    macro_environment: str = "Stable"
) -> Dict[str, Any]:
    """
    Computes a multi-path financial balance forecast and returns a typed payload.
    Bypasses LangChain tool wrapping and ephemeral chart_cache database writes.
    """
    real_balance = forecaster_engine.fetch_live_balance(account_id, user_uuid)
    S0, mu, _ = forecaster_engine.fetch_and_calculate_parameters(account_id, real_balance, user_uuid, 60)
    
    df_temp, timeline = forecaster_engine.run_hybrid_simulation(
        account_id=account_id,
        S0=S0,
        mu=mu,
        user_uuid=user_uuid,
        days=days,
        paths=1000000,
        discipline_multiplier=discipline_multiplier,
        drift_adjustment=drift_adjustment,
        stress_test_active=stress_test_active,
        macro_environment=macro_environment
    )

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

    return {
        "series": [{"bank_name": account_id, "data": bank_data}],
        "timeline": timeline
    }


def compute_expense_forecast(
    user_uuid: str,
    account_id: str,
    days: int = 30,
    discipline_multiplier: float = 1.0,
    drift_adjustment: float = 0.0,
    stress_test_active: bool = False,
    macro_environment: str = "Stable"
) -> Dict[str, Any]:
    """
    Computes an expense velocity simulation and returns cumulative projections.
    """
    current_balance = forecaster_engine.fetch_live_balance(account_id, user_uuid)
    E0, mu_E = forecaster_engine.fetch_expense_parameters(account_id, user_uuid, 60)
    df_temp, timeline = forecaster_engine.run_expense_simulation(
        account_id=account_id,
        E0=E0,
        mu=mu_E,
        user_uuid=user_uuid,
        days=days,
        paths=1000000,
        discipline_multiplier=discipline_multiplier,
        drift_adjustment=drift_adjustment,
        stress_test_active=stress_test_active,
        macro_environment=macro_environment,
        current_balance=current_balance
    )

    bank_data = []
    if not df_temp.empty:
        exp_vals = df_temp.iloc[0].values.tolist()
        for i in range(days + 1):
            bank_data.append({
                "Day": f"Day {i}",
                "Projected Spend": round(exp_vals[i], 2)
            })

    return {
        "series": [{"bank_name": account_id, "data": bank_data}],
        "timeline": timeline
    }
