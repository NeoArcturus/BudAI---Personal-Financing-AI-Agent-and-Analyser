import pandas as pd
import os
from services.api_integrator.get_account_detail import UserAccount


class ExpenseAnalysis:
    def __init__(self, identifier=None, user_uuid=None):
        self.user_account = UserAccount(identifier, user_uuid)
        if self.user_account.needs_user_clarification:
            raise ValueError("MULTIPLE_ACCOUNTS")
        self.classified_data = None
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.abspath(
            os.path.join(self.base_dir, '..', '..'))
        self.csv_dir = os.path.join(self.root_dir, "saved_media", "csvs")
        os.makedirs(self.csv_dir, exist_ok=True)

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

    def export_weekly_spend_data(self):
        if self.classified_data is None or self.classified_data.empty:
            return None
        self.classified_data['Date'] = pd.to_datetime(
            self.classified_data['Date'], format='ISO8601')
        self.classified_data.set_index('Date', inplace=True)
        weekly_spend = self.classified_data['Amount'].resample(
            'W').sum().reset_index()
        csv_path = os.path.join(
            self.csv_dir, f"weekly_spend_{self.user_account.account_id}.csv")
        weekly_spend.to_csv(csv_path, index=False)
        return csv_path

    def export_monthly_spend_data(self):
        if self.classified_data is None or self.classified_data.empty:
            return None
        self.classified_data['Date'] = pd.to_datetime(
            self.classified_data['Date'], format='ISO8601')
        self.classified_data.set_index('Date', inplace=True)
        monthly_spend = self.classified_data['Amount'].resample(
            'M').sum().reset_index()
        csv_path = os.path.join(
            self.csv_dir, f"monthly_spend_{self.user_account.account_id}.csv")
        monthly_spend.to_csv(csv_path, index=False)
        return csv_path

    def export_daily_spend_data(self):
        if self.classified_data is None or self.classified_data.empty:
            return None
        self.classified_data['Date'] = pd.to_datetime(
            self.classified_data['Date'], format='ISO8601')
        self.classified_data.set_index('Date', inplace=True)
        daily_spend = self.classified_data['Amount'].resample(
            'D').sum().reset_index()
        csv_path = os.path.join(
            self.csv_dir, f"daily_spend_{self.user_account.account_id}.csv")
        daily_spend.to_csv(csv_path, index=False)
        return csv_path
