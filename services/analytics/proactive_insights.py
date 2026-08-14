import json
from datetime import datetime, timedelta
from config import SessionLocal
from models.database_models import User, ProactiveInsight
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

def generate_proactive_insights_for_all_users():
    """
    Runs in the background (e.g. nightly) to evaluate user data and generate 
    Proactive Insights.
    """
    logger.info(json.dumps({"message": f"Starting Proactive Insights generation job.", "status_code": 200}))
    try:
        with SessionLocal() as session:
            users = session.query(User).all()
            for user in users:
                # Stub: In a real scenario, this would run the forecast simulation 
                # and call the LLM to generate insights. 
                # We'll create a dummy insight to demonstrate the pipeline.
                
                # Check if they already have a recent insight to avoid spam
                recent = session.query(ProactiveInsight).filter(
                    ProactiveInsight.user_uuid == user.user_uuid,
                    ProactiveInsight.created_at >= datetime.utcnow() - timedelta(hours=12)
                ).first()
                
                if not recent:
                    new_insight = ProactiveInsight(
                        user_uuid=user.user_uuid,
                        insight_text="Your grocery spending is tracking 15% higher than last month. Consider reviewing your upcoming subscriptions to offset the cost.",
                        insight_type="warning",
                        urgency_level=3
                    )
                    session.add(new_insight)
                    
            session.commit()
        logger.info(json.dumps({"message": f"Proactive Insights generation completed.", "status_code": 200}))
    except Exception as e:
        logger.error(json.dumps({"message": f"Error generating proactive insights: {e}", "status_code": 500}))
