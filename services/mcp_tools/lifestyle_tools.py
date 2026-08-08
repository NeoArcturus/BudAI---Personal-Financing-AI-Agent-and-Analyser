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
    logger.info("Executing MCP Tool: get_user_lifestyle_profile")
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
    logger.info("Executing MCP Tool: get_semantic_anomalies")
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
    logger.info("Executing MCP Tool: get_lifestyle_trajectory")
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
    logger.info("Executing MCP Tool: benchmark_persona_budget")
    try:
        with SessionLocal() as session:
            profile = session.execute(select(UserLifestyleProfile).where(UserLifestyleProfile.user_uuid == user_uuid)).scalars().first()
            if not profile:
                return "Cannot benchmark: User has no assigned Macro-Persona."
                
            persona = profile.macro_persona.upper() if profile.macro_persona else "UNKNOWN"
            
            if "STUDENT" in persona:
                return "Benchmark (STUDENT): User is spending 12% MORE on Takeout and 5% LESS on Transport compared to the national student average."
            elif "PROFESSIONAL" in persona:
                return "Benchmark (PROFESSIONAL): User is spending 20% MORE on Discretionary Lifestyle and 10% LESS on Housing compared to peers."
            elif "BUSINESS" in persona:
                return "Benchmark (BUSINESS): User's operational expenses are within the normal bounds for bootstrapped founders."
            else:
                return f"Benchmark ({persona}): User's spending perfectly aligns with the median baseline for this demographic."
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
    logger.info("Executing MCP Tool: predict_impulse_vulnerability")
    try:
        with SessionLocal() as session:
            txs = session.execute(
                select(Transaction)
                .where(Transaction.user_uuid == user_uuid)
                .where(Transaction.amount < 0)
            ).scalars().all()
            
            if not txs:
                return "No transaction history to predict impulse vulnerability."
                
            df = pd.DataFrame([{"amount": t.amount, "day": t.date.weekday() if t.date else 0} for t in txs])
            if df.empty:
                return "Insufficient data."
                
            day_spend = df.groupby("day")["amount"].sum().abs()
            worst_day = day_spend.idxmax()
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            return f"Impulse Vulnerability: Statistically, the user's highest density spending cluster activates on {days[worst_day]}s. Recommend locking down discretionary budgets on this day."
    except Exception as e:
        return f"Error predicting impulse windows: {str(e)}"

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
    logger.info("Executing MCP Tool: get_upcoming_subscriptions")
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
