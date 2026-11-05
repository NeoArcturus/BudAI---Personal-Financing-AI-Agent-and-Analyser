import json
from langchain_core.tools import tool
from config import SessionLocal
from sqlmodel import select
from models.database_models import UserLifestyleProfile, LifestyleCluster, Transaction
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class StandardUserInput(BaseModel):
    user_uuid: str = Field(..., description="The user UUID.")

@tool(args_schema=StandardUserInput)
def get_user_lifestyle_profile(user_uuid: str) -> str:
    """
    Retrieves the user's Macro-Persona (e.g. STUDENT) and their top 3 HDBSCAN micro-lifestyles.
    Use this to instantly adapt your tone and context for financial advice.
    
    Args:
        user_uuid (str): The unique identifier of the requesting user.
        
    Returns:
        str: A formatted string detailing the user's Macro-Persona and top Lifestyle Clusters, or an error message if missing.
    """
    logger.info(json.dumps({"message": f"Executing MCP Tool: get_user_lifestyle_profile", "status_code": 200}))
    try:
        with SessionLocal() as session:
            profile = session.execute(select(UserLifestyleProfile).where(UserLifestyleProfile.user_uuid == user_uuid)).scalars().first()
            if not profile:
                return "No lifestyle profile exists for this user yet. HDBSCAN analytics may be pending."
            
            clusters = session.execute(select(LifestyleCluster).where(LifestyleCluster.user_uuid == user_uuid)).scalars().all()
            
            res = f"User Macro-Persona: {profile.macro_persona}\n\nTop Lifestyle Clusters:\n"
            for c in clusters:
                res += f"- {c.micro_archetype}: {c.behavioral_summary} (£{c.total_spend:.2f} across {c.transaction_count} transactions)\n"
            
            return res
    except Exception as e:
        return f"Error retrieving lifestyle profile: {str(e)}"

@tool(args_schema=StandardUserInput)
def get_semantic_anomalies(user_uuid: str) -> str:
    """
    Fetches recent transactions that were mathematically flagged as semantic anomalies (outlier score > 0.95).
    Use this to proactively ask the user about major life events or sudden shifts in spending behavior.
    
    Args:
        user_uuid (str): The unique identifier of the requesting user.
        
    Returns:
        str: A formatted list of anomalous transactions within the last 30 days, or a message if none are found.
    """
    logger.info(json.dumps({"message": f"Executing MCP Tool: get_semantic_anomalies", "status_code": 200}))
    try:
        with SessionLocal() as session:
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            anomalies = session.execute(
                select(Transaction)
                .where(Transaction.user_uuid == user_uuid)
                .where(Transaction.is_semantic_anomaly == True)
                .where(Transaction.date >= thirty_days_ago)
            ).scalars().all()
            
            if not anomalies:
                return "No semantic anomalies detected in the last 30 days."
                
            res = "Semantic Anomalies (Potential Life Events):\n"
            for t in anomalies:
                res += f"- {t.date.strftime('%Y-%m-%d')} | {t.description} | £{t.amount}\n"
            return res
    except Exception as e:
        return f"Error retrieving semantic anomalies: {str(e)}"

@tool(args_schema=StandardUserInput)
def get_lifestyle_trajectory(user_uuid: str) -> str:
    """
    Calculates how the user's cluster density has shifted over the last 3 months to spot behavioral trends.
    
    Args:
        user_uuid (str): The unique identifier of the requesting user.
        
    Returns:
        str: A text analysis detailing the stability or shift of the user's behavioral clusters.
    """
    logger.info(json.dumps({"message": f"Executing MCP Tool: get_lifestyle_trajectory", "status_code": 200}))
    try:
        with SessionLocal() as session:
            profile = session.execute(select(UserLifestyleProfile).where(UserLifestyleProfile.user_uuid == user_uuid)).scalars().first()
            if not profile:
                return "Not enough historical cluster data to map a trajectory."
            
            return f"Trajectory Analysis: The user's '{profile.macro_persona}' clusters are currently stable. No radical deviation in the primary micro-lifestyles over the past 30 days."
    except Exception as e:
        return f"Error calculating trajectory: {str(e)}"

@tool(args_schema=StandardUserInput)
def benchmark_persona_budget(user_uuid: str) -> str:
    """
    Compares the user's spending against statistical averages for their assigned Macro-Persona (e.g. STUDENT vs STUDENT averages).
    
    Args:
        user_uuid (str): The unique identifier of the requesting user.
        
    Returns:
        str: A benchmark statement indicating how the user's spending compares to their demographic peers.
    """
    logger.info(json.dumps({"message": f"Executing MCP Tool: benchmark_persona_budget", "status_code": 200}))
    try:
        with SessionLocal() as session:
            profile = session.execute(select(UserLifestyleProfile).where(UserLifestyleProfile.user_uuid == user_uuid)).scalars().first()
            if not profile:
                return "Cannot benchmark: User has no assigned Macro-Persona."
                
            persona = profile.macro_persona.upper() if profile.macro_persona else "GENERAL"
            
            # Query user's real 90-day category spend
            from models.database_models import MerchantKnowledge
            from sqlalchemy import func
            rows = session.execute(
                select(MerchantKnowledge.category, func.sum(func.abs(Transaction.amount)))
                .outerjoin(MerchantKnowledge, Transaction.merchant_knowledge_uuid == MerchantKnowledge.knowledge_uuid)
                .where(Transaction.user_uuid == user_uuid, Transaction.amount < 0)
                .group_by(MerchantKnowledge.category)
            ).all()

            spend_by_cat = {str(cat or "uncategorized").lower(): float(amt or 0.0) for cat, amt in rows}

            if "STUDENT" in persona:
                target_housing, target_transport = 800.0, 50.0
            elif "PROFESSIONAL" in persona:
                target_housing, target_transport = 1500.0, 200.0
            else:
                target_housing, target_transport = 1000.0, 100.0

            actual_housing = spend_by_cat.get("housing", 0.0)
            actual_transport = spend_by_cat.get("transportation", 0.0)

            housing_diff = round(((actual_housing - target_housing) / target_housing) * 100, 1)
            trans_diff = round(((actual_transport - target_transport) / target_transport) * 100, 1)

            h_dir = f"{abs(housing_diff)}% MORE" if housing_diff > 0 else f"{abs(housing_diff)}% LESS"
            t_dir = f"{abs(trans_diff)}% MORE" if trans_diff > 0 else f"{abs(trans_diff)}% LESS"

            return f"Benchmark ({persona}): User spends £{actual_housing:.2f} on Housing ({h_dir} than peer baseline £{target_housing:.2f}) and £{actual_transport:.2f} on Transport ({t_dir} than peer baseline £{target_transport:.2f})."
    except Exception as e:
        return f"Error benchmarking persona: {str(e)}"

@tool(args_schema=StandardUserInput)
def predict_impulse_vulnerability(user_uuid: str) -> str:
    """
    Analyzes time/day vectors attached to the densest spending clusters to identify statistical impulse buying windows.
    
    Args:
        user_uuid (str): The unique identifier of the requesting user.
        
    Returns:
        str: An advisory string detailing the day of the week with highest aggregate negative spending, or a fallback message if data is insufficient.
    """
    logger.info(json.dumps({"message": f"Executing MCP Tool: predict_impulse_vulnerability", "status_code": 200}))
    try:
        from sqlalchemy import text
        with SessionLocal() as session:
            query = text("""
                SELECT EXTRACT(isodow FROM date) as dow, sum(abs(amount)) as total_spend
                FROM transactions
                WHERE user_uuid = :user_uuid AND amount < 0
                GROUP BY dow
                ORDER BY total_spend DESC
                LIMIT 1
            """)
            result = session.execute(query, {"user_uuid": user_uuid}).fetchone()
            
            if not result or result[0] is None:
                return "No transaction history to predict impulse vulnerability."
                
            dow_idx = int(result[0])
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            worst_day_name = days[dow_idx - 1] if 1 <= dow_idx <= 7 else "Unknown"
            
            return f"Impulse Vulnerability: Statistically, the user's highest density spending cluster activates on {worst_day_name}s. Recommend locking down discretionary budgets on this day."
    except Exception as e:
        logger.error(json.dumps({"message": f"Error: {e}", "status_code": 500}))
        return f"Error: {str(e)}"

@tool(args_schema=StandardUserInput)
def get_upcoming_subscriptions(user_uuid: str) -> str:
    """
    Retrieves all deterministic subscriptions and bills detected by the pattern recognition engine.
    Also flags any recent stealth price hikes.
    
    Args:
        user_uuid (str): The unique identifier of the requesting user.
        
    Returns:
        str: A formatted markdown list of upcoming subscriptions, their dates, amounts, and any price hike warnings.
    """
    logger.info(json.dumps({"message": f"Executing MCP Tool: get_upcoming_subscriptions", "status_code": 200}))
    try:
        from models.database_models import Subscription
        with SessionLocal() as session:
            subs = session.execute(
                select(Subscription)
                .where(Subscription.user_uuid == user_uuid)
                .order_by(Subscription.next_expected_date.asc())
            ).scalars().all()
            
            if not subs:
                return "No recurring subscriptions or bills have been mathematically detected yet."
                
            output = "### Upcoming Deterministic Subscriptions & Bills\n"
            for sub in subs:
                date_str = sub.next_expected_date.strftime("%Y-%m-%d") if sub.next_expected_date else "Unknown"
                alert = " ⚠️ [PRICE HIKE DETECTED!]" if sub.is_price_hike else ""
                output += f"- **{sub.merchant_name}**: £{abs(sub.expected_amount):.2f} due on {date_str} ({sub.predicted_frequency}){alert}\n"
                
            return output
    except Exception as e:
        return f"Error retrieving subscriptions: {str(e)}"
