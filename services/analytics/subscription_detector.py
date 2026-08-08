import numpy as np
from datetime import datetime, timedelta
import uuid
import re
from sqlmodel import select
from config import SessionLocal
from models.database_models import Transaction, Subscription
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class SubscriptionDetector:
    def __init__(self, time_variance_threshold_days=15.0, price_hike_threshold=1.05):
        self.time_variance_threshold_days = time_variance_threshold_days
        self.price_hike_threshold = price_hike_threshold

    def analyze_user_subscriptions(self, user_uuid: str):
        logger.info(f"Starting subscription detection for user {user_uuid}")
        
        with SessionLocal() as session:
            txs = session.execute(
                select(Transaction)
                .where(Transaction.user_uuid == user_uuid)
                .where(Transaction.amount < 0)
                .where(Transaction.category == "Bills & Utilities")
                .order_by(Transaction.date.asc())
            ).scalars().all()
            
            if not txs:
                logger.info(f"No bill-related transactions found for user {user_uuid}")
                return
                
                

            anti_grocery_keywords = [
                "tesco", "sainsburys", "asda", "morrisons", "aldi", "lidl", 
                "waitrose", "iceland", "co-op", "deliveroo", "uber eats", 
                "just eat", "mcdonalds", "kfc", "dominos", "food"
            ]
            
            merchant_groups = {}
            
            for tx in txs:
                if not tx.description or not tx.date:
                    continue
                    
                raw_desc_lower = tx.description.lower()
                
                if any(keyword in raw_desc_lower for keyword in anti_grocery_keywords):
                    continue
                if "sent money" in raw_desc_lower:
                    continue
                    
                clean_desc = re.sub(r'[^a-zA-Z\s]', '', raw_desc_lower).strip()
                clean_desc = re.sub(r'\s+', ' ', clean_desc)
                
                if not clean_desc:
                    continue
                    
                if clean_desc not in merchant_groups:
                    merchant_groups[clean_desc] = []
                merchant_groups[clean_desc].append(tx)

            detected_subscriptions = []
            
            for merchant_key, m_txs in merchant_groups.items():
                if len(m_txs) < 3:
                    continue
                    
                dates = [tx.date for tx in m_txs]
                amounts = [abs(tx.amount) for tx in m_txs]
                
                time_gaps = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
                median_gap = np.median(time_gaps)
                
                if median_gap < 5:
                    continue
                    
                std_gap = np.std(time_gaps)
                
                if std_gap <= self.time_variance_threshold_days:
                    freq = "Monthly"
                    if 6 <= median_gap <= 8:
                        freq = "Weekly"
                    elif 12 <= median_gap <= 16:
                        freq = "Bi-Weekly"
                    elif 80 <= median_gap <= 100:
                        freq = "Quarterly"
                    elif 110 <= median_gap <= 130:
                        freq = "Termly"
                    elif 350 <= median_gap <= 380:
                        freq = "Annually"
                        
                    last_date = dates[-1]
                    next_date = last_date + timedelta(days=int(np.mean(time_gaps)))
                    
                    latest_amount = amounts[-1]
                    historical_avg = np.mean(amounts[:-1])
                    
                    is_hike = latest_amount > (historical_avg * self.price_hike_threshold)
                    
                    pretty_merchant_name = merchant_key.title()
                    
                    last_tx = m_txs[-1]
                    
                    detected_subscriptions.append({
                        "merchant_name": pretty_merchant_name,
                        "bank_uuid": getattr(last_tx, "bank_uuid", None),
                        "expected_amount": latest_amount,
                        "last_payment_date": last_date,
                        "last_payment_amount": latest_amount,
                        "predicted_frequency": freq,
                        "next_expected_date": next_date,
                        "is_price_hike": is_hike
                    })
                    
            session.query(Subscription).filter(Subscription.user_uuid == user_uuid).delete()
            
            if detected_subscriptions:
                for sub in detected_subscriptions:
                    new_sub = Subscription(
                        subscription_uuid=str(uuid.uuid4()),
                        user_uuid=user_uuid,
                        merchant_name=sub["merchant_name"],
                        bank_uuid=sub["bank_uuid"],
                        expected_amount=sub["expected_amount"],
                        last_payment_date=sub["last_payment_date"],
                        last_payment_amount=sub["last_payment_amount"],
                        predicted_frequency=sub["predicted_frequency"],
                        next_expected_date=sub["next_expected_date"],
                        is_price_hike=sub["is_price_hike"]
                    )
                    session.add(new_sub)
                session.commit()
                logger.info(f"Detected and saved {len(detected_subscriptions)} subscriptions for user {user_uuid}")
            else:
                session.commit()
                logger.info(f"No valid subscriptions found for user {user_uuid}")
