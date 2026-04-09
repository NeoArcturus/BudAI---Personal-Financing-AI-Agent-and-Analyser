import os
import pandas as pd
import numpy as np
from sqlalchemy import text
from config import SessionLocal
from services.api_integrator.get_account_detail import UserAccounts
from services.Forecaster_Agent.mathematics.mathematics import run_hybrid_engine, run_converged_expense_engine


class ForecasterAgent:
    def __init__(self, db_path=None):
        pass

    def fetch_live_balance(self, identifier, user_uuid):
        if str(identifier).upper() == "ALL" or "," in identifier:
            raise ValueError(
                "ForecasterAgent strictly handles a single account identifier.")

        user_acc = UserAccounts(user_id=user_uuid)
        balance = user_acc.get_account_balance(identifier, user_uuid)
        return float(balance) if balance is not None else 0.0

    def fetch_and_calculate_parameters(self, account_id, current_balance, user_uuid, lookback_days=60):
        try:
            with SessionLocal() as session:
                row = session.execute(text("""SELECT b.bank_name 
                               FROM banks b 
                               LEFT JOIN accounts a
                               ON b.bank_uuid = a.bank_uuid
                               WHERE a.account_id=:account_id"""), {"account_id": account_id}).fetchone()

                user = UserAccounts(user_uuid)
                bank_name = row[0] if row else account_id
                df = user.get_transactions(
                    bank_name_or_id=bank_name, user_uuid=user_uuid)
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            return current_balance, -0.01, 0.05

        if 'timestamp' in df.columns and 'date' not in df.columns:
            df['date'] = df['timestamp']
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
        mu = float(np.clip(mu, -0.02, 0.02))

        sigma = recent_data['Returns'].std()

        if np.isnan(mu):
            mu = -0.01
        if np.isnan(sigma) or sigma == 0:
            sigma = 0.05
        return current_balance, mu, sigma

    def fetch_expense_parameters(self, account_id, user_uuid, lookback_days=60):
        try:
            with SessionLocal() as session:
                row = session.execute(text("""SELECT b.bank_name 
                               FROM banks b 
                               LEFT JOIN accounts a
                               ON b.bank_uuid = a.bank_uuid
                               WHERE a.account_id=:account_id"""), {"account_id": account_id}).fetchone()

                user = UserAccounts(user_uuid)
                bank_name = row[0] if row else account_id
                df = user.get_transactions(
                    bank_name_or_id=bank_name, user_uuid=user_uuid)
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            return 50.0, 0.001

        if 'timestamp' in df.columns and 'date' not in df.columns:
            df['date'] = df['timestamp']
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

        mu_E = float(np.clip(mu_E, -0.02, 0.02))
        return E0, mu_E

    def run_hybrid_simulation(self, account_id, S0, mu, days=60, paths=1000):
        df = run_hybrid_engine(S0, mu, days, paths, account_id)
        return df

    def run_expense_simulation(self, account_id, E0, mu, days=30, paths=1000):
        df = run_converged_expense_engine(E0, mu, days, paths, account_id)
        return df
