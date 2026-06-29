import pandas as pd
import logging
from services.api_integrator.get_account_detail import UserAccounts
from config import SessionLocal
from models.database_models import Transaction
from sqlalchemy import text
from datetime import datetime
from services.logger_setup import get_core_logger
logger = get_core_logger(__name__)

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
        try:
            df = self.user_account.get_bank_transactions(
                self.identifier, self.user_uuid, from_date, to_date, expense_only=True)
            if df.empty:
                return False
            time_col = 'date' if 'date' in df.columns else 'timestamp'
            amt_col = 'amount' if 'amount' in df.columns else 'Amount'
            if time_col not in df.columns or amt_col not in df.columns:
                return False
            df.rename(columns={time_col: 'Date',
                      amt_col: 'Amount'}, inplace=True)
            df = df.loc[:, ~df.columns.duplicated()].copy()
            df['Date'] = pd.to_datetime(
                df['Date'], errors='coerce', utc=True)
            mask = (df['Date'] >= pd.to_datetime(from_date, utc=True)) & (
                df['Date'] <= pd.to_datetime(to_date, format='ISO8601', utc=True))
            df = df.loc[mask].copy()
            df['Amount'] = df['Amount'].abs()
            if not df.empty:
                cols_to_keep = ['Date', 'Amount']
                if 'bank_name' in df.columns:
                    cols_to_keep.append('bank_name')
                self.classified_data = df[cols_to_keep].copy()
                return True
            return False
        except Exception:
            logger.error("An error occurred in this block", exc_info=True)
            return False
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
