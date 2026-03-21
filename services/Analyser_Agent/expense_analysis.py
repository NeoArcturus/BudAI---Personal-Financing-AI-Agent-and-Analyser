import pandas as pd
from services.api_integrator.get_account_detail import UserAccounts


class ExpenseAnalysis:
    def __init__(self, identifier=None, user_uuid=None):
        if str(identifier).upper() == "ALL" or "," in str(identifier):
            raise ValueError(
                "ExpenseAnalysis strictly handles a single account identifier.")

        self.identifier = identifier
        self.user_uuid = user_uuid
        self.user_account = UserAccounts(user_id=user_uuid)
        self.classified_data = None

    def fetch_data(self, from_date, to_date):
        df = self.user_account.get_transactions(
            self.identifier, self.user_uuid, from_date, to_date)
        if df is None or df.empty:
            return False

        time_col = next((c for c in df.columns if c.lower()
                        in ['timestamp', 'date']), None)
        amt_col = next(
            (c for c in df.columns if c.lower() in ['amount']), None)

        if not time_col or not amt_col:
            return False

        df.rename(columns={time_col: 'Date', amt_col: 'Amount'}, inplace=True)

        df = df.loc[:, ~df.columns.duplicated()].copy()

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)

        mask = (df['Date'] >= pd.to_datetime(from_date, utc=True)) & (
            df['Date'] <= pd.to_datetime(to_date, utc=True))
        df = df.loc[mask]

        df = df[df['Amount'] < 0].copy()
        df['Amount'] = df['Amount'].abs()

        if df.empty:
            return False

        cols_to_keep = ['Date', 'Amount']
        if 'bank_name' in df.columns:
            cols_to_keep.append('bank_name')

        self.classified_data = df[cols_to_keep].copy()
        return True

    def _get_pivoted_data(self, freq_str):
        df = self.classified_data.copy()
        df.set_index('Date', inplace=True)
        resampled = df['Amount'].resample(freq_str).sum().reset_index()
        resampled['Date'] = resampled['Date'].dt.date
        return resampled

    def get_daily_spend_data(self):
        return self._get_pivoted_data('D')

    def get_weekly_spend_data(self):
        return self._get_pivoted_data('W-MON')

    def get_monthly_spend_data(self):
        return self._get_pivoted_data('ME')
