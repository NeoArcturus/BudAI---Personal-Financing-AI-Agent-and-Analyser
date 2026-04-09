import pandas as pd
from services.api_integrator.get_account_detail import UserAccounts
from config import SessionLocal
from models.database_models import Transaction
from sqlalchemy import text
from datetime import datetime


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
            db_transactions = []
            with SessionLocal() as session:
                query = text("""
                    SELECT t.transaction_uuid, t.user_uuid, t.bank_uuid, t.account_id,
                           t.date, t.amount, t.category, t.description
                    FROM transactions t
                    JOIN accounts a ON t.account_id = a.account_id
                    JOIN banks b ON a.bank_uuid = b.bank_uuid
                    WHERE t.user_uuid = :user_uuid
                      AND (b.bank_name = :identifier OR a.account_id = :identifier)
                      AND t.date >= :from_date
                      AND t.date <= :to_date
                """)
                rows = session.execute(
                    query, {
                        "user_uuid": self.user_uuid,
                        "identifier": self.identifier,
                        "from_date": from_date,
                        "to_date": to_date
                    }
                ).fetchall()

                print(
                    "[BACKEND LOG] Checking if the database has the transactions or not")
                if rows:
                    print(
                        f"[BACKEND LOG] Database has transactions between {from_date} and {to_date}")
                    for row in rows:
                        db_transactions.append({
                            "transaction_uuid": row[0],
                            "user_uuid": row[1],
                            "bank_uuid": row[2],
                            "account_id": row[3],
                            "date": row[4],
                            "amount": row[5],
                            "category": row[6],
                            "description": row[7]
                        })

                    if db_transactions:
                        df = pd.DataFrame(db_transactions)
                        time_col = next((c for c in df.columns if c.lower()
                                        in ['timestamp', 'date']), None)
                        amt_col = next(
                            (c for c in df.columns if c.lower() in ['amount']), None)

                        if time_col and amt_col:
                            df.rename(
                                columns={time_col: 'Date', amt_col: 'Amount'}, inplace=True)

                            df = df.loc[:, ~df.columns.duplicated()].copy()

                            df['Date'] = pd.to_datetime(
                                df['Date'], errors='coerce', utc=True)

                            mask = (df['Date'] >= pd.to_datetime(from_date, utc=True)) & (
                                df['Date'] <= pd.to_datetime(to_date, utc=True))
                            df = df.loc[mask]

                            df = df[df['Amount'] < 0].copy()
                            df['Amount'] = df['Amount'].abs()

                            if not df.empty:
                                cols_to_keep = ['Date', 'Amount']
                                if 'bank_name' in df.columns:
                                    cols_to_keep.append('bank_name')

                                self.classified_data = df[cols_to_keep].copy()
                                return True

            if not db_transactions:
                print(
                    f"[BACKEND LOG] Database does not have transactions between {from_date} and {to_date}")
                print("Checking TrueLayer API...")
                df = self.user_account.get_bank_transactions(
                    self.identifier, self.user_uuid, from_date, to_date)
                if df is None or df.empty:
                    print(
                        "[BACKEND LOG] Cannot find transactions from TrueLayer API!")
                    return False

                print(
                    f"[BACKEND LOG] Number of transactions returned: {len(df)}")

                time_col = next((c for c in df.columns if c.lower()
                                in ['timestamp', 'date']), None)
                amt_col = next(
                    (c for c in df.columns if c.lower() in ['amount']), None)

                if not time_col or not amt_col:
                    return False

                df.rename(columns={time_col: 'Date',
                          amt_col: 'Amount'}, inplace=True)

                df = df.loc[:, ~df.columns.duplicated()].copy()

                df['Date'] = pd.to_datetime(
                    df['Date'], errors='coerce', utc=True)

                mask = (df['Date'] >= pd.to_datetime(from_date, utc=True)) & (
                    df['Date'] <= pd.to_datetime(to_date, utc=True))
                df = df.loc[mask]

                df = df[df['Amount'] < 0].copy()
                df['Amount'] = df['Amount'].abs()

                if not df.empty:
                    cols_to_keep = ['Date', 'Amount']
                    if 'bank_name' in df.columns:
                        cols_to_keep.append('bank_name')

                    self.classified_data = df[cols_to_keep].copy()
                    return True

                return False

            return True
        except Exception:
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
