import sqlite3
import pandas as pd
import numpy as np
from services.api_integrator.get_account_detail import UserAccounts


class FinancialHealthAnalyzer:
    def __init__(self, user_uuid, db_path="budai_memory.db"):
        self.user_uuid = user_uuid
        self.db_path = db_path

    def _fetch_transactions(self):
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM transactions WHERE user_uuid = ?"
            try:
                df = pd.read_sql_query(query, conn, params=(self.user_uuid,))
                if df.empty:
                    return df
                df.columns = [c.lower() for c in df.columns]
                if 'timestamp' in df.columns and 'date' not in df.columns:
                    df['date'] = df['timestamp']
                return df
            except Exception:
                return pd.DataFrame()

    def _fetch_total_liquidity(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT bank_name FROM banks WHERE user_uuid = ?", (self.user_uuid,))
                banks = cursor.fetchall()
            total = 0.0
            for b in banks:
                try:
                    balance = UserAccounts(b[0], self.user_uuid).get_account_balance(
                        b[0], self.user_uuid)
                    if balance is not None:
                        total += float(balance)
                except Exception:
                    pass
            return float(total)
        except Exception:
            return 0.0

    def calculate_subsistence_floor(self):
        df = self._fetch_transactions()
        if df.empty or 'category' not in df.columns:
            return 0.0

        df['date'] = pd.to_datetime(df['date'], format='ISO8601', utc=True)
        inelastic_categories = ['Rent', 'Mortgage',
                                'Utilities', 'Insurance', 'Groceries', 'Debt_Min']

        subsistence_df = df[df['category'].isin(
            inelastic_categories) & (df['amount'] < 0)].copy()
        subsistence_df['amount'] = subsistence_df['amount'].abs()

        if subsistence_df.empty:
            return 0.0

        monthly_totals = subsistence_df.groupby(
            pd.Grouper(key='date', freq='ME'))['amount'].sum()
        return float(monthly_totals.tail(3).mean()) if not monthly_totals.empty else 0.0

    def calculate_liquid_runway(self):
        liquidity = self._fetch_total_liquidity()
        floor = self.calculate_subsistence_floor()
        if floor <= 0:
            return float('inf')
        mean_daily_floor = floor / 30.0
        return float(liquidity / mean_daily_floor)

    def avalanche_debt_optimization(self, monthly_surplus=0.0):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT account_name, balance, interest_rate, min_payment FROM liabilities WHERE user_uuid = ?", (self.user_uuid,))
                debts = cursor.fetchall()
        except sqlite3.OperationalError:
            return []

        sorted_debts = sorted(debts, key=lambda x: x[2], reverse=True)
        plan = []
        remaining_surplus = monthly_surplus

        for name, principal, rate, min_pmt in sorted_debts:
            monthly_interest = (principal * (rate / 100)) / 12
            payment = min_pmt + remaining_surplus
            remaining_surplus = 0
            plan.append({
                "target": name,
                "payment_allocation": payment,
                "interest_saved_annually": monthly_interest * 12
            })
        return plan

    def calculate_net_worth_velocity(self):
        df = self._fetch_transactions()
        if df.empty:
            return 0.0

        df['date'] = pd.to_datetime(df['date'], format='ISO8601', utc=True)
        df['amount'] = df['amount'].astype(float)
        monthly_net = df.groupby(pd.Grouper(key='date', freq='ME'))[
            'amount'].sum()

        if len(monthly_net) < 2:
            return float(monthly_net.sum())

        return float(monthly_net.diff().mean())

    def calculate_mpc(self):
        df = self._fetch_transactions()
        if df.empty:
            return 0.0

        df['date'] = pd.to_datetime(df['date'], format='ISO8601', utc=True)
        income_df = df[df['amount'] > 0]
        expense_df = df[df['amount'] < 0].copy()
        expense_df['amount'] = expense_df['amount'].abs()

        monthly_income = income_df.groupby(
            pd.Grouper(key='date', freq='ME'))['amount'].sum()
        monthly_expense = expense_df.groupby(
            pd.Grouper(key='date', freq='ME'))['amount'].sum()

        merged = pd.DataFrame(
            {'income': monthly_income, 'expense': monthly_expense}).fillna(0)
        merged['delta_income'] = merged['income'].diff()
        merged['delta_expense'] = merged['expense'].diff()

        valid_months = merged[merged['delta_income'] > 0]
        if valid_months.empty:
            return 0.0

        mpc = (valid_months['delta_expense'] /
               valid_months['delta_income']).mean()
        return float(max(0.0, min(mpc, 1.0)))

    def calculate_shock_absorption(self):
        df = self._fetch_transactions()
        if df.empty:
            return 0.0

        liquidity = self._fetch_total_liquidity()
        df['date'] = pd.to_datetime(df['date'], format='ISO8601', utc=True)
        monthly_net = df.groupby(pd.Grouper(key='date', freq='ME'))[
            'amount'].sum()
        max_deficit = abs(monthly_net.min()) if monthly_net.min() < 0 else 0

        if max_deficit == 0:
            return float('inf')
        return float(liquidity / max_deficit)

    def calculate_interest_drag(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT SUM(balance * (interest_rate / 100) / 12) FROM liabilities WHERE user_uuid = ?", (self.user_uuid,))
                res = cursor.fetchone()
                monthly_interest = float(res[0]) if res and res[0] else 0.0
        except sqlite3.OperationalError:
            monthly_interest = 0.0

        df = self._fetch_transactions()
        if df.empty:
            return 0.0

        df['date'] = pd.to_datetime(df['date'], format='ISO8601', utc=True)
        income_df = df[df['amount'] > 0]
        if income_df.empty:
            return 0.0

        avg_monthly_income = income_df.groupby(pd.Grouper(key='date', freq='ME'))[
            'amount'].sum().mean()
        if avg_monthly_income == 0:
            return 0.0

        return float((monthly_interest / avg_monthly_income) * 100)
