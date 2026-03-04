import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import matplotlib
from api_integrator.get_account_detail import UserAccount
from Forecaster_Agent.mathematics.mathematics import run_hybrid_engine, run_converged_expense_engine

matplotlib.use('Agg')


class ForecasterAgent:
    def __init__(self, db_path="budai_memory.db"):
        self.db_path = db_path
        self.user_acc = None

    def fetch_live_balance(self, identifier):
        self.user_acc = UserAccount(identifier)
        balance_data = self.user_acc.get_account_balance()
        if isinstance(balance_data, list) and len(balance_data) > 0:
            return float(balance_data[0].get("available", balance_data[0].get("current", 0.0)))
        return 0.0

    def fetch_and_calculate_parameters(self, current_balance, lookback_days=60):
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT date, amount FROM transactions"
                df = pd.read_sql_query(query, conn)
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

    def fetch_expense_parameters(self, lookback_days=60):
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT date, amount FROM transactions WHERE amount < 0"
                df = pd.read_sql_query(query, conn)
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

    def analyze_and_plot_expenses(self, csv_path, E0, mu_E, days=30):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        img_dir = os.path.join(os.path.abspath(
            os.path.join(current_dir, '..')), "saved_media", "images")
        os.makedirs(img_dir, exist_ok=True)
        if not os.path.exists(csv_path):
            return
        acc_id = self.user_acc.account_id if self.user_acc and self.user_acc.account_id else "default"
        df = pd.read_csv(csv_path, header=None)
        converged_path = df.iloc[0].values
        t_days = np.arange(0, days + 1)
        expected_path = E0 * np.exp(mu_E * t_days)
        plt.figure(figsize=(15, 8))
        plt.plot(converged_path, color="#e74c3c",
                 label="Converged Stochastic Path", linewidth=2)
        plt.plot(expected_path, color="#2c3e50",
                 label="Historical Baseline", linestyle="--", linewidth=3)
        plt.title(f'BudAI Expense Forecast Convergence ({days} Days)')
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.savefig(os.path.join(
            img_dir, f"expense_convergence_path_{acc_id}.png"), dpi=300)
        plt.close()

    def analyze_and_plot(self, csv_path, S0, threshold_pct=0.2):
        risk_threshold = S0 * threshold_pct
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, '..'))
        img_dir = os.path.join(root_dir, "saved_media", "images")
        os.makedirs(img_dir, exist_ok=True)

        acc_id = self.user_acc.account_id if self.user_acc and self.user_acc.account_id else "default"
        plot_path = os.path.join(
            img_dir, f"monte_carlo_forecast_paths_{acc_id}.png")

        if not os.path.exists(csv_path):
            return

        paths_df = pd.read_csv(csv_path, header=None)

        if len(paths_df) > 10:
            final_balances = paths_df.iloc[:, -1]
            p5_val = final_balances.quantile(0.05, interpolation='nearest')
            path1_idx = (final_balances - p5_val).abs().idxmin()
            p95_val = final_balances.quantile(0.95, interpolation='nearest')
            path3_idx = (final_balances - p95_val).abs().idxmin()

            path1 = paths_df.iloc[path1_idx]
            path2 = paths_df.mean(axis=0)
            path3 = paths_df.iloc[path3_idx]
            paths_df = pd.DataFrame([path1, path2, path3])

        narratives = [
            {"label": "Path 1: Careless Scenario (5th Percentile)",
             "color": "#e74c3c", "ls": "-"},
            {"label": "Path 2: Expected Mean Trajectory",
                "color": "#7f8c8d", "ls": "--"},
            {"label": "Path 3: Optimal Budgeting (Stable)",
             "color": "#27ae60", "ls": "-"}
        ]

        plt.figure(figsize=(15, 8))
        for i, meta in enumerate(narratives):
            if i < len(paths_df):
                plt.plot(paths_df.iloc[i], color=meta["color"], label=meta["label"],
                         linewidth=3 if i != 1 else 2, linestyle=meta["ls"])

        plt.axhline(y=risk_threshold, color='black', linestyle=':',
                    alpha=0.6, label=f'Risk Threshold (£{risk_threshold:.2f})')
        plt.title(
            f'BudAI Narrative Forecast (Risk at {threshold_pct*100}% of Balance)', fontsize=16, pad=20)
        plt.xlabel('Days into the Future', fontsize=12)
        plt.ylabel('Projected Balance (£)', fontsize=12)

        max_val = paths_df.max().max()
        plt.ylim(0, max(max_val * 1.2, risk_threshold * 1.5))
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.legend(loc='upper left', frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
