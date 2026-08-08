import calendar
from datetime import datetime
from sqlalchemy.orm import Session
from models.database_models import Budget, Transaction
from config import SessionLocal
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class BudgetEngine:
    def __init__(self, user_uuid: str):
        self.user_uuid = user_uuid

    def get_variance_for_category(self, category: str = None) -> list[dict]:
        """
        Calculates Spend Velocity, Projected Spend, and Variance for user budgets.
        If category is None, calculates for all active budgets.
        """
        results = []
        now = datetime.utcnow()
        current_day = now.day
        _, total_days = calendar.monthrange(now.year, now.month)
        
        start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        with SessionLocal() as session:
            query = session.query(Budget).filter(Budget.user_uuid == self.user_uuid)
            if category:
                query = query.filter(Budget.category.ilike(category))
            budgets = query.all()
            
            if not budgets:
                return []
                
            for budget in budgets:
                spent_query = session.query(Transaction).filter(
                    Transaction.user_uuid == self.user_uuid,
                    Transaction.category.ilike(budget.category),
                    Transaction.date >= start_of_month,
                    Transaction.date <= now,
                    Transaction.amount < 0
                ).all()
                
                spent_so_far = sum(abs(tx.amount) for tx in spent_query)
                
                limit = budget.monthly_limit
                
                elapsed_days = max(1, current_day)
                
                if limit > 0:
                    velocity = (spent_so_far / limit) / (elapsed_days / total_days)
                else:
                    velocity = 0.0
                    
                projected_spend = (spent_so_far / elapsed_days) * total_days
                
                variance = limit - projected_spend
                
                results.append({
                    "category": budget.category,
                    "monthly_limit": limit,
                    "spent_so_far": round(spent_so_far, 2),
                    "velocity": round(velocity, 2),
                    "projected_spend": round(projected_spend, 2),
                    "variance": round(variance, 2)
                })
                
        return results
