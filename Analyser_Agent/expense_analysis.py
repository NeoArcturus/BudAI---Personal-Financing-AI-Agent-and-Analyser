import matplotlib.pyplot as mp
import pandas as pd
import os
from api_integrator.get_account_detail import UserAccount


class ExpenseAnalysis:
    def __init__(self):
        self.user_account = UserAccount()
        self.classified_data = None
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.abspath(os.path.join(self.base_dir, '..'))
        self.img_dir = os.path.join(self.root_dir, "saved_media", "images")
        os.makedirs(self.img_dir, exist_ok=True)

    def fetch_data(self, from_date, to_date):
        df = self.user_account.all_transactions(from_date, to_date)

        if df is None or df.empty:
            return False

        if 'timestamp' in df.columns and 'Date' not in df.columns:
            df.rename(columns={'timestamp': 'Date'}, inplace=True)
        elif 'date' in df.columns and 'Date' not in df.columns:
            df.rename(columns={'date': 'Date'}, inplace=True)

        if 'amount' in df.columns and 'Amount' not in df.columns:
            df.rename(columns={'amount': 'Amount'}, inplace=True)

        df = df[df['Amount'] < 0].copy()
        df['Amount'] = df['Amount'].abs()

        if df.empty:
            return False

        self.classified_data = df
        return True

    def plot_weekly_spend(self):
        if self.classified_data is None or self.classified_data.empty:
            return
        self.classified_data['Date'] = pd.to_datetime(
            self.classified_data['Date'], format='ISO8601')
        self.classified_data.set_index('Date', inplace=True)
        weekly_spend = self.classified_data['Amount'].resample('W').sum()
        mp.figure(figsize=(10, 6))
        mp.plot(weekly_spend.index, weekly_spend.values, marker='o')
        mp.title('Weekly Spend Over Time')
        mp.xlabel('Week')
        mp.ylabel('Total Spend')
        mp.xticks(rotation=45)
        mp.tight_layout()
        mp.savefig(os.path.join(self.img_dir, "expense_plot_weekly.png"))
        mp.close()

    def plot_monthly_spend(self):
        if self.classified_data is None or self.classified_data.empty:
            return
        self.classified_data['Date'] = pd.to_datetime(
            self.classified_data['Date'], format='ISO8601')
        self.classified_data.set_index('Date', inplace=True)
        monthly_spend = self.classified_data['Amount'].resample('M').sum()
        mp.figure(figsize=(10, 6))
        mp.plot(monthly_spend.index, monthly_spend.values, marker='o')
        mp.title('Monthly Spend Over Time')
        mp.xlabel('Month')
        mp.ylabel('Total Spend')
        mp.xticks(rotation=45)
        mp.tight_layout()
        mp.savefig(os.path.join(self.img_dir, "expense_plot_monthly.png"))
        mp.close()

    def plot_daily_spend(self):
        if self.classified_data is None or self.classified_data.empty:
            return
        self.classified_data['Date'] = pd.to_datetime(
            self.classified_data['Date'], format='ISO8601')
        self.classified_data.set_index('Date', inplace=True)
        daily_spend = self.classified_data['Amount'].resample('D').sum()
        mp.figure(figsize=(10, 6))
        mp.plot(daily_spend.index, daily_spend.values, marker='o')
        mp.title('Daily Spend Over Time')
        mp.xlabel('Days')
        mp.ylabel('Total Spend')
        mp.xticks(rotation=45)
        mp.tight_layout()
        mp.savefig(os.path.join(self.img_dir, "expense_plot_daily.png"))
        mp.close()
