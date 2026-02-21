import os
import webbrowser
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from dotenv import load_dotenv
from Categorizer_Agent.api_integrator.access_token_generator import AccessTokenGenerator
from Categorizer_Agent.api_integrator.get_account_detail import UserAccount


class ForecasterAgent:
    def __init__(self, db_path="budai_memory.db"):
        self.db_path = db_path
        self.token_gen = AccessTokenGenerator()
        self.user_acc = None

    def _authenticate(self):
        if not self.token_gen.regenerate_auth_token_using_refresh_token():
            webbrowser.open(self.token_gen.get_auth_link())
            self.token_gen.app.run(port=8080)

    def fetch_live_balance(self):
        self._authenticate()
        load_dotenv(override=True)
        self.user_acc = UserAccount()
        balance_data = self.user_acc.get_account_balance()
        if isinstance(balance_data, list) and len(balance_data) > 0:
            return float(balance_data[0].get("available", balance_data[0].get("current", 0.0)))
        return 0.0

    def fetch_and_calculate_parameters(self, current_balance, lookback_days=60):
        with sqlite3.connect(self.db_path) as conn:
            query = f"SELECT date, amount FROM transactions"
            df = pd.read_sql_query(query, conn)

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

    def run_cpp_simulation(self, S0, mu, sigma, days=30, paths=5000):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cpp_file = os.path.join(current_dir, "forecaster.cpp")
        executable = os.path.join(current_dir, "forecaster")
        subprocess.run(["g++", "-O3", "-o", executable, cpp_file], check=True)
        subprocess.run([executable, str(S0), str(mu), str(sigma), str(
            days), str(paths)], cwd=current_dir, check=True)

    def analyze_and_plot(self, risk_threshold=100.0):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "all_paths.csv")
        plot_path = os.path.join(
            current_dir, "monte_carlo_student_forecast.png")

        if not os.path.exists(csv_path):
            return

        paths_df = pd.read_csv(csv_path, header=None)

        plt.figure(figsize=(20, 10))
        for i in range(len(paths_df)):
            plt.plot(paths_df.iloc[i], color='blue')

        mean_path = paths_df.mean(axis=0)
        plt.plot(mean_path, color='red', linewidth=3,
                 label='Mean Forecasted Balance')
        plt.axhline(y=risk_threshold, color='red', linestyle='--', alpha=0.5)

        max_val = paths_df.max().max()
        plt.ylim(0, max(max_val * 1.1, risk_threshold * 1.1))

        plt.title('Monte Carlo GBM Forecast')
        plt.xlabel('Days into the Future')
        plt.ylabel('Projected Balance (£)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)


if __name__ == "__main__":
    agent = ForecasterAgent()
    real_balance = agent.fetch_live_balance()
    S0, mu, sigma = agent.fetch_and_calculate_parameters(real_balance, 60)
    agent.run_cpp_simulation(S0, mu, sigma, days=60, paths=1000)
    agent.analyze_and_plot(risk_threshold=100.0)
