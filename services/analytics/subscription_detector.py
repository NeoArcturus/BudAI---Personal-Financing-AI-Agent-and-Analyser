import json
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
        logger.info(json.dumps({"message": f"Starting subscription detection for user {user_uuid}", "status_code": 200}))
        
        with SessionLocal() as session:
            txs = session.execute(
                select(Transaction)
                .where(Transaction.user_uuid == user_uuid)
                .where(Transaction.amount < 0)
                .where(Transaction.category.in_([
                    "Subscriptions & Digital Services", 
                    "Utilities", 
                    "Entertainment & Lifestyle",
                    "Healthcare"
                ]))
                .order_by(Transaction.date.asc())
            ).scalars().all()
            
            if not txs:
                logger.info(json.dumps({"message": f"No bill-related transactions found for user {user_uuid}", "status_code": 200}))
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
                    
                group_key = (tx.account_id, clean_desc)
                if group_key not in merchant_groups:
                    merchant_groups[group_key] = []
                merchant_groups[group_key].append(tx)

            detected_subscriptions = []
            
            for group_key, m_txs in merchant_groups.items():
                account_id, merchant_key = group_key
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
                    
                    # 90-day expiration time check
                    from models.status_codes import PipelineStatus
                    now_utc = datetime.utcnow().replace(tzinfo=None)
                    days_overdue = (now_utc - next_date.replace(tzinfo=None)).days
                    
                    status_val = "303-410" if days_overdue > 90 else PipelineStatus.SUBSCRIPTION_DETECTED.value
                    if is_hike and days_overdue <= 90:
                        status_val = PipelineStatus.SUBSCRIPTION_PRICE_HIKE.value

                    # Prevent Duplication & Overwriting (UPSERT logic)
                    existing_sub = session.query(Subscription).filter_by(
                        user_uuid=user_uuid, 
                        merchant_name=pretty_merchant_name,
                        account_id=account_id
                    ).first()

                    if existing_sub:
                        existing_sub.last_payment_date = last_date.replace(tzinfo=None)
                        existing_sub.last_payment_amount = float(latest_amount)
                        existing_sub.next_expected_date = next_date.replace(tzinfo=None)
                        existing_sub.is_price_hike = bool(is_hike)
                        existing_sub.status = status_val
                        existing_sub.last_updated = datetime.utcnow()
                        detected_subscriptions.append(existing_sub)
                    else:
                        new_sub = Subscription(
                            subscription_uuid=str(uuid.uuid4()),
                            user_uuid=user_uuid,
                            merchant_name=pretty_merchant_name,
                            account_id=account_id,
                            bank_uuid=getattr(last_tx, "bank_uuid", None),
                            expected_amount=float(latest_amount),
                            last_payment_date=last_date.replace(tzinfo=None),
                            last_payment_amount=float(latest_amount),
                            predicted_frequency=freq,
                            next_expected_date=next_date.replace(tzinfo=None),
                            is_price_hike=bool(is_hike),
                            status=status_val
                        )
                        session.add(new_sub)
                        detected_subscriptions.append(new_sub)
                    
                    # Update tags for these transactions
                    for tx in m_txs:
                        existing_tags = tx.tags or []
                        if "#recurring" not in existing_tags:
                            existing_tags.append("#recurring")
                        if is_hike and tx.transaction_uuid == last_tx.transaction_uuid:
                            if "#price-hike" not in existing_tags:
                                existing_tags.append("#price-hike")
                        tx.tags = existing_tags
                        
            if detected_subscriptions:
                session.commit()
                logger.info(json.dumps({"message": f"Detected and saved {len(detected_subscriptions)} subscriptions for user {user_uuid}", "status_code": 200}))
            else:
                session.commit()
                logger.info(json.dumps({"message": f"No valid subscriptions found for user {user_uuid}", "status_code": 200}))
