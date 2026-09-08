import json
import logging
from sqlalchemy import text
from config import SessionLocal
from services.api_integrator.account_reader import AccountReader
from services.logger_setup import get_core_logger
logger = get_core_logger(__name__)

class FinancialHealthAnalyzer:
    def __init__(self, user_uuid, db_path=None):
        self.user_uuid = user_uuid
        # We no longer load all transactions into a Pandas DataFrame!
        # self.df = self._fetch_transactions()  <-- REMOVED

    def _fetch_total_liquidity(self):
        try:
            with SessionLocal() as session:
                banks = session.execute(
                    text("SELECT bank_name FROM banks WHERE user_uuid = :user_uuid"), 
                    {"user_uuid": self.user_uuid}
                ).fetchall()
            total = 0.0
            for b in banks:
                try:
                    balance = AccountReader(self.user_uuid).get_account_balance(
                        b[0], self.user_uuid)
                    if balance is not None:
                        total += float(balance)
                except Exception:
                    logger.error(json.dumps({"message": f"An error occurred fetching balance for {b[0]}", "status_code": 500}), exc_info=True)
                    pass
            return float(total)
        except Exception:
            logger.error(json.dumps({"message": "An error occurred fetching banks", "status_code": 500}), exc_info=True)
            return 0.0

    def calculate_subsistence_floor(self):
        query = text("""
            SELECT sum(abs(t.amount)) as monthly_floor
            FROM transactions t
            LEFT JOIN merchant_knowledge m ON t.merchant_knowledge_uuid = m.knowledge_uuid
            WHERE t.user_uuid = :user_uuid
              AND m.category IN ('Rent', 'Mortgage', 'Utilities', 'Insurance', 'Groceries', 'Debt_Min')
              AND t.amount < 0
            GROUP BY time_bucket('1 month', t.date)
            ORDER BY time_bucket('1 month', t.date) DESC
            LIMIT 3
        """)
        with SessionLocal() as session:
            results = session.execute(query, {"user_uuid": self.user_uuid}).fetchall()
            if not results:
                return 0.0
            totals = [float(r[0]) for r in results if r[0] is not None]
            if not totals: return 0.0
            return sum(totals) / len(totals)

    def calculate_liquid_runway(self):
        liquidity = self._fetch_total_liquidity()
        floor = self.calculate_subsistence_floor()
        if floor <= 0:
            return float('inf')
        mean_daily_floor = floor / 30.0
        return float(liquidity / mean_daily_floor)

    def avalanche_debt_optimization(self, monthly_surplus=0.0):
        # TODO: The 'liabilities' table does not exist in the current database schema.
        # Returning an empty plan until the schema supports explicit liability tracking.
        return []

    def calculate_net_worth_velocity(self):
        query = text("""
            WITH monthly_net AS (
                SELECT time_bucket('1 month', date) as month, sum(amount) as net
                FROM transactions
                WHERE user_uuid = :user_uuid
                GROUP BY time_bucket('1 month', date)
                ORDER BY month ASC
            ),
            diffs AS (
                SELECT net - lag(net) OVER (ORDER BY month) as delta
                FROM monthly_net
            )
            SELECT avg(delta) FROM diffs;
        """)
        with SessionLocal() as session:
            res = session.execute(query, {"user_uuid": self.user_uuid}).fetchone()
            if res and res[0] is not None:
                return float(res[0])
            
            # Fallback if no diffs (only 1 month of data)
            fallback = session.execute(text("""
                SELECT sum(amount) FROM transactions WHERE user_uuid = :user_uuid
            """), {"user_uuid": self.user_uuid}).fetchone()
            return float(fallback[0]) if fallback and fallback[0] else 0.0

    def calculate_mpc(self):
        query = text("""
            WITH monthly_flows AS (
                SELECT time_bucket('1 month', date) as month,
                       sum(CASE WHEN amount > 0 THEN amount ELSE 0 END) as income,
                       sum(CASE WHEN amount < 0 THEN abs(amount) ELSE 0 END) as expense
                FROM transactions
                WHERE user_uuid = :user_uuid
                GROUP BY time_bucket('1 month', date)
                ORDER BY month ASC
            ),
            deltas AS (
                SELECT 
                    income - lag(income) OVER (ORDER BY month) as delta_income,
                    expense - lag(expense) OVER (ORDER BY month) as delta_expense
                FROM monthly_flows
            )
            SELECT avg(delta_expense / delta_income)
            FROM deltas
            WHERE delta_income > 0;
        """)
        with SessionLocal() as session:
            res = session.execute(query, {"user_uuid": self.user_uuid}).fetchone()
            mpc = float(res[0]) if res and res[0] is not None else 0.0
            return float(max(0.0, min(mpc, 1.0)))

    def calculate_shock_absorption(self):
        liquidity = self._fetch_total_liquidity()
        query = text("""
            SELECT min(net) FROM (
                SELECT sum(amount) as net
                FROM transactions
                WHERE user_uuid = :user_uuid
                GROUP BY time_bucket('1 month', date)
            ) t;
        """)
        with SessionLocal() as session:
            res = session.execute(query, {"user_uuid": self.user_uuid}).fetchone()
            max_deficit = abs(float(res[0])) if res and res[0] is not None and float(res[0]) < 0 else 0.0
            
        if max_deficit == 0:
            return float('inf')
        return float(liquidity / max_deficit)

    def calculate_interest_drag(self):
        # TODO: The 'liabilities' table and 'interest_rate' column do not currently exist in the DB schema.
        # Returning 0.0 until the schema is updated to support credit card / loan interest rates.
        monthly_interest = 0.0

        if monthly_interest == 0.0:
            return 0.0

        query = text("""
            SELECT avg(income) FROM (
                SELECT sum(amount) as income
                FROM transactions
                WHERE user_uuid = :user_uuid AND amount > 0
                GROUP BY time_bucket('1 month', date)
            ) t;
        """)
        with SessionLocal() as session:
            res = session.execute(query, {"user_uuid": self.user_uuid}).fetchone()
            avg_monthly_income = float(res[0]) if res and res[0] is not None else 0.0
            
        if avg_monthly_income == 0:
            return 0.0
        return float((monthly_interest / avg_monthly_income) * 100)
