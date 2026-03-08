import os
import pandas as pd
import numpy as np
import sqlite3
from services.api_integrator.get_account_detail import UserAccount
from services.Forecaster_Agent.mathematics.mathematics import run_hybrid_engine, run_converged_expense_engine


class ForecasterAgent:
    def __init__(self, db_path="budai_memory.db"):
        self.db_path = db_path
        self.user_acc = None

    def fetch_live_balance(self, identifier, user_uuid):
        self.user_acc = UserAccount(identifier, user_uuid)
        if self.user_acc.needs_user_clarification:
            raise ValueError("MULTIPLE_ACCOUNTS")
        balance_data = self.user_acc.get_account_balance()
        if isinstance(balance_data, list) and len(balance_data) > 0:
            return float(balance_data[0].get("available", balance_data[0].get("current", 0.0)))
        return 0.0

    def fetch_and_calculate_parameters(self, current_balance, user_uuid, lookback_days=60):
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT date, amount FROM transactions WHERE user_uuid = ?"
                df = pd.read_sql_query(query, conn, params=(user_uuid,))
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            return current_balance, -0.01, 0.05

        df['Date'] = pd.to_datetime(
            df['date'], format='ISO8601', utc=True).dt.date
        daily_net = df.groupby('Date')['amount'].sum(
        ).reset_index().sort_values('Date')
        daily_net['Reverse_Amount'] = daily_net['amount'].iloc[::-1]
        historical_balances = [current_balance]
        temp_balance = current_balance
        for amt in daily_net['Reverse_Amount'].values[:-1]:
            temp_balance -= amt
            historical_balances.append(temp_balance)
        daily_net['Balance'] = historical_balances[::-1]
        recent_data = daily_net.tail(lookback_days).copy()
        recent_data['Safe_Balance'] = recent_data['Balance'].apply(
            lambda x: max(x, 10))
        recent_data['Returns'] = np.log(
            recent_data['Safe_Balance'] / recent_data['Safe_Balance'].shift(1))
        mu = recent_data['Returns'].mean()
        sigma = recent_data['Returns'].std()
        if np.isnan(mu):
            mu = -0.01
        if np.isnan(sigma) or sigma == 0:
            sigma = 0.05
        return current_balance, mu, sigma

    def fetch_expense_parameters(self, user_uuid, lookback_days=60):
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT date, amount FROM transactions WHERE amount < 0 AND user_uuid = ?"
                df = pd.read_sql_query(query, conn, params=(user_uuid,))
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            return 50.0, 0.001

        df['Date'] = pd.to_datetime(
            df['date'], format='ISO8601', utc=True).dt.date
        df['amount'] = df['amount'].abs()
        daily_expenses = df.groupby(
            'Date')['amount'].sum().reset_index().sort_values('Date')
        recent_data = daily_expenses.tail(lookback_days).copy()
        recent_data['Safe_Amount'] = recent_data['amount'].apply(
            lambda x: max(x, 1))
        E0 = recent_data['Safe_Amount'].mean()
        recent_data['Returns'] = np.log(
            recent_data['Safe_Amount'] / recent_data['Safe_Amount'].shift(1))
        mu_E = recent_data['Returns'].mean()
        if np.isnan(mu_E):
            mu_E = 0.001
        return E0, mu_E

    def run_hybrid_simulation(self, account_id, S0, mu, days=30, paths=1000000):
        return run_hybrid_engine(S0, mu, days, paths, str(account_id))

    def run_expense_simulation(self, account_id, E0, mu_E, days=30, paths=1000000):
        return run_converged_expense_engine(E0, mu_E, days, paths, str(account_id))
