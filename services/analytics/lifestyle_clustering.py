import os
import re
import uuid
import json
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
from sqlalchemy import text
from sqlmodel import select

from config import SessionLocal
from models.database_models import UserLifestyleProfile, LifestyleCluster
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

ESSENTIAL_CATEGORIES = {
    "housing", "rent", "mortgage", "utilities", "bills",
    "bills & utilities", "groceries", "supermarket",
    "transport", "public transport", "healthcare", "insurance", "council tax"
}


class ClusterProfile(BaseModel):
    hdbscan_cluster_id: int
    micro_archetype: str
    behavioral_summary: str
    total_spend: float
    transaction_count: int


class LifestyleAnalysisOutput(BaseModel):
    macro_persona: str
    discretionary_ratio: float
    weekend_ratio: float
    late_night_ratio: float
    clusters: List[ClusterProfile]


class LifestyleClusteringService:
    """
    Deterministic Lifestyle Topology & Behavioral Profiling Engine.
    Replaces opaque density clustering (HDBSCAN) with transparent,
    reproducible arithmetic (Needs vs. Wants ratio, Pareto 80/20 merchant
    concentration, and weekly/temporal skew).
    """

    def _clean_merchant_name(self, text_val: str) -> str:
        if not text_val:
            return "Unknown"
        t = text_val.strip()
        t = re.sub(r'card transaction of .*? issued by', '', t, flags=re.IGNORECASE)
        t = re.sub(r'card transaction of .*? to', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\(fee:.*?\)', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\b(gbp|usd|eur)\b', '', t, flags=re.IGNORECASE)
        t = re.sub(r'[0-9]{4,}', '', t)
        cleaned = t.strip(" -_#*")
        return cleaned.title() if cleaned else "Unknown"

    def analyze_user_lifestyle(self, user_uuid: str) -> Optional[Dict[str, Any]]:
        """
        Calculates lifestyle topology in memory:
        1. Needs vs. Wants ratio (Essential vs. Discretionary)
        2. Pareto 80/20 Merchant Concentration
        3. Weekly Rhythm (Weekend vs. Weekday velocity)
        4. Late-Night Timing Vulnerability (21:00 - 02:00)
        """
        logger.info(json.dumps({
            "message": f"Starting deterministic lifestyle analysis for user {user_uuid}",
            "status_code": 200
        }))

        with SessionLocal() as session:
            query = text("""
                SELECT 
                    t.transaction_uuid,
                    COALESCE(NULLIF(TRIM(t.semi_cleaned_description), ''), NULLIF(TRIM(t.fully_cleaned_description), ''), TRIM(t.description)) as merchant,
                    COALESCE(LOWER(m.category), 'uncategorized') as cat,
                    COALESCE(LOWER(t.sub_category), '') as sub_cat,
                    ABS(t.amount) as spend,
                    t.date
                FROM transactions t
                LEFT JOIN merchant_knowledge m ON t.merchant_knowledge_uuid = m.knowledge_uuid
                WHERE t.user_uuid = :user_uuid AND t.amount < 0
                ORDER BY t.date DESC
            """)
            rows = session.execute(query, {"user_uuid": user_uuid}).fetchall()

            if not rows or len(rows) < 3:
                logger.info(json.dumps({
                    "message": f"Insufficient transactions ({len(rows) if rows else 0}) for lifestyle profiling",
                    "status_code": 200
                }))
                return None

            total_spend = 0.0
            essential_spend = 0.0
            discretionary_spend = 0.0
            essential_count = 0
            discretionary_count = 0

            weekend_spend = 0.0
            weekend_count = 0

            late_night_spend = 0.0
            late_night_count = 0

            merchant_spend_map: Dict[str, float] = {}
            merchant_count_map: Dict[str, int] = {}
            discretionary_merchants: Dict[str, float] = {}

            all_tx_ids = []

            for r in rows:
                tx_id, raw_merchant, cat, sub_cat, spend, tx_date = r
                spend_val = float(spend or 0.0)
                total_spend += spend_val
                all_tx_ids.append(str(tx_id))

                clean_merchant = self._clean_merchant_name(raw_merchant)

                # 1. Essential vs. Discretionary
                is_essential = (cat in ESSENTIAL_CATEGORIES) or (sub_cat in ESSENTIAL_CATEGORIES)
                if is_essential:
                    essential_spend += spend_val
                    essential_count += 1
                else:
                    discretionary_spend += spend_val
                    discretionary_count += 1
                    discretionary_merchants[clean_merchant] = discretionary_merchants.get(clean_merchant, 0.0) + spend_val

                merchant_spend_map[clean_merchant] = merchant_spend_map.get(clean_merchant, 0.0) + spend_val
                merchant_count_map[clean_merchant] = merchant_count_map.get(clean_merchant, 0) + 1

                # 2. Weekend Skew (Saturday=5, Sunday=6)
                if tx_date:
                    dow = tx_date.weekday()
                    if dow in (5, 6):
                        weekend_spend += spend_val
                        weekend_count += 1

                # 3. Late-Night Timing (21:00 - 02:00)
                ts = tx_date
                if ts and hasattr(ts, "hour"):
                    if ts.hour >= 21 or ts.hour < 2:
                        late_night_spend += spend_val
                        late_night_count += 1

            if total_spend <= 0.0:
                return None

            # Calculate mathematical ratios
            discretionary_ratio = round(discretionary_spend / max(essential_spend, 1.0), 2)
            weekend_ratio = round(weekend_spend / max(total_spend, 1.0), 2)
            late_night_ratio = round(late_night_spend / max(discretionary_spend, 1.0), 2) if discretionary_spend > 0 else 0.0

            # 4. Pareto 80/20 Merchant Concentration on Discretionary Cash
            sorted_merchants = sorted(discretionary_merchants.items(), key=lambda x: x[1], reverse=True)
            cumulative = 0.0
            pareto_merchants = []
            target_80_pct = discretionary_spend * 0.80

            for m_name, m_amt in sorted_merchants:
                cumulative += m_amt
                pareto_merchants.append((m_name, m_amt))
                if cumulative >= target_80_pct or len(pareto_merchants) >= 4:
                    break

            top_merchants_str = ", ".join([f"{name} (£{amt:.0f})" for name, amt in pareto_merchants[:3]]) if pareto_merchants else "Varied merchants"

            # 5. Deterministic Macro-Persona Selection
            if discretionary_ratio > 1.1 and weekend_ratio >= 0.40:
                macro_persona = "Weekend Spender"
            elif discretionary_ratio > 1.2:
                macro_persona = "Lifestyle Expander"
            elif discretionary_ratio < 0.40:
                macro_persona = "Frugal Essentialist"
            elif late_night_ratio > 0.20:
                macro_persona = "Late-Night Consumer"
            else:
                macro_persona = "Balanced Living"

            # 6. Structured Clusters
            clusters: List[ClusterProfile] = []

            # Cluster 1: Needs vs. Wants Split
            clusters.append(ClusterProfile(
                hdbscan_cluster_id=1,
                micro_archetype="Needs vs. Wants Allocation",
                behavioral_summary=f"£{discretionary_spend:.2f} discretionary spending vs £{essential_spend:.2f} survival essentials ({discretionary_ratio}x multiplier).",
                total_spend=round(discretionary_spend, 2),
                transaction_count=discretionary_count
            ))

            # Cluster 2: Pareto Top Merchants
            clusters.append(ClusterProfile(
                hdbscan_cluster_id=2,
                micro_archetype="Dominant Merchants",
                behavioral_summary=f"80% of discretionary outflow flows through {len(pareto_merchants)} core destinations: {top_merchants_str}.",
                total_spend=round(sum(amt for _, amt in pareto_merchants), 2),
                transaction_count=sum(merchant_count_map.get(m, 1) for m, _ in pareto_merchants)
            ))

            # Cluster 3: Weekly Rhythm
            weekend_pct = int(weekend_ratio * 100)
            clusters.append(ClusterProfile(
                hdbscan_cluster_id=3,
                micro_archetype="Weekly Rhythm",
                behavioral_summary=f"{weekend_pct}% of total spending occurs on Saturday and Sunday (normal baseline is 28%).",
                total_spend=round(weekend_spend, 2),
                transaction_count=weekend_count
            ))

            # Cluster 4: Late-Night Check (if present)
            if late_night_ratio > 0.10:
                night_pct = int(late_night_ratio * 100)
                clusters.append(ClusterProfile(
                    hdbscan_cluster_id=4,
                    micro_archetype="Evening Timing",
                    behavioral_summary=f"{night_pct}% of non-essential orders occur between 9 PM and 2 AM (£{late_night_spend:.2f}).",
                    total_spend=round(late_night_spend, 2),
                    transaction_count=late_night_count
                ))

            # Hash check for caching
            current_hash = hashlib.sha256(f"{user_uuid}_{macro_persona}_{total_spend:.2f}_{len(rows)}".encode("utf-8")).hexdigest()

            # Persist to database
            profile = session.execute(
                select(UserLifestyleProfile).where(UserLifestyleProfile.user_uuid == user_uuid)
            ).scalars().first()

            if not profile:
                profile = UserLifestyleProfile(
                    profile_uuid=str(uuid.uuid4()),
                    user_uuid=user_uuid,
                    macro_persona=macro_persona,
                    last_cluster_hash=current_hash,
                    last_updated=datetime.utcnow()
                )
                session.add(profile)
            else:
                profile.macro_persona = macro_persona
                profile.last_cluster_hash = current_hash
                profile.last_updated = datetime.utcnow()

            # Refresh cluster table
            session.execute(text("DELETE FROM lifestyle_clusters WHERE user_uuid = :uuid"), {"uuid": user_uuid})

            for c in clusters:
                lc = LifestyleCluster(
                    cluster_uuid=str(uuid.uuid4()),
                    user_uuid=user_uuid,
                    hdbscan_cluster_id=c.hdbscan_cluster_id,
                    micro_archetype=c.micro_archetype,
                    behavioral_summary=c.behavioral_summary,
                    total_spend=c.total_spend,
                    transaction_count=c.transaction_count,
                    last_updated=datetime.utcnow()
                )
                session.add(lc)

            session.commit()

            logger.info(json.dumps({
                "message": f"Deterministic lifestyle analysis completed for {user_uuid}: {macro_persona} (ratio={discretionary_ratio}, weekend={weekend_ratio})",
                "status_code": 200
            }))

            return {
                "macro_persona": macro_persona,
                "discretionary_ratio": discretionary_ratio,
                "weekend_ratio": weekend_ratio,
                "late_night_ratio": late_night_ratio,
                "essential_spend": essential_spend,
                "discretionary_spend": discretionary_spend,
                "top_merchants": [m for m, _ in pareto_merchants],
                "clusters": [c.model_dump() for c in clusters]
            }
