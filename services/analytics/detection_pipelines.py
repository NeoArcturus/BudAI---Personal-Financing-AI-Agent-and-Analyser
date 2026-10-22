import json
import os
from sqlmodel import Session
from sqlalchemy import text
from services.logger_setup import get_core_logger, log_pipeline_data
from langchain_openai import OpenAIEmbeddings

logger = get_core_logger(__name__)

def run_detection_pipelines(session: Session, user_uuid: str) -> dict:
    """
    Step 2 of Phase 3: Mathematical & Vector Detection
    Runs SQL window functions and Qdrant vector search to find Subscriptions and Utilities.
    """
    
    # 1. Temporal Math Query using LAG and STDDEV
    query = text("""
    WITH lagged_txs AS (
        SELECT 
            semi_cleaned_description as merchant,
            amount,
            date,
            LAG(date) OVER (PARTITION BY semi_cleaned_description ORDER BY date) as prev_date
        FROM transactions
        WHERE user_uuid = :user_uuid
    ),
    date_diffs AS (
        SELECT 
            merchant,
            amount,
            EXTRACT(DAY FROM (date - prev_date)) as diff_days
        FROM lagged_txs
        WHERE prev_date IS NOT NULL
    ),
    stats AS (
        SELECT 
            merchant,
            COUNT(*) as tx_count,
            AVG(diff_days) as avg_days,
            COALESCE(STDDEV(diff_days), 0) as stddev_days,
            COALESCE(STDDEV(amount), 0) as stddev_amount,
            AVG(amount) as avg_amount
        FROM date_diffs
        GROUP BY merchant
        HAVING COUNT(*) >= 2
    )
    SELECT * FROM stats;
    """)
    
    results = session.execute(query, {"user_uuid": user_uuid}).fetchall()
    
    subscriptions = []
    potential_utilities = []
    
    # 2. Calculating Variance & Subscription Split
    for row in results:
        merchant = row.merchant
        std_days = float(row.stddev_days)
        std_amount = float(row.stddev_amount)
        avg_amount = float(row.avg_amount)
        
        # Subscription: Tight time rhythm (<3 days variance), exact amount (variance ~0)
        if std_days < 3.0 and std_amount < 0.01:
            subscriptions.append({"merchant": merchant, "amount": avg_amount, "frequency_days": row.avg_days})
            
        # Utility: Relaxed time rhythm (<5 days variance), variable amount
        elif std_days < 5.0 and std_amount >= 0.01:
            potential_utilities.append({"merchant": merchant, "avg_amount": avg_amount, "frequency_days": row.avg_days})
            
    # 3. Vector Verification for Utilities via Qdrant
    verified_utilities = []
    if potential_utilities:
        try:
            from services.embedding_service import get_embedding_service
            from services.qdrant_service import get_qdrant_service
            from qdrant_client import models
            
            emb_service = get_embedding_service()
            qdrant = get_qdrant_service()
            
            merchants_to_embed = [u["merchant"] for u in potential_utilities if u["merchant"]]
            if merchants_to_embed:
                vectors = emb_service.embed_documents(merchants_to_embed)
                
                utility_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="category",
                            match=models.MatchValue(value="Utility")
                        )
                    ]
                )
                
                for idx, u in enumerate(potential_utilities):
                    if not u["merchant"]: 
                        continue
                    v = vectors[idx]
                    hits = qdrant.search_vectors(
                        "merchant_knowledge",
                        query_vector=v,
                        limit=1,
                        query_filter=utility_filter
                    )
                    # High semantic match (score >= 0.85)
                    if hits and hits[0].get("score", 0.0) >= 0.85:
                        verified_utilities.append(u)
                        
        except Exception as e:
            logger.error(f"Vector search failed in detection pipeline: {e}", exc_info=True)
            
    output_data = {
        "new_subscriptions": subscriptions,
        "new_utilities": verified_utilities
    }

    log_pipeline_data(
        logger=logger,
        pipeline="detection_pipelines",
        stage="detection_complete",
        data={
            "user_uuid": user_uuid,
            "subscriptions_count": len(subscriptions),
            "utilities_count": len(verified_utilities),
            "subscriptions": subscriptions,
            "utilities": verified_utilities
        },
        message=f"Detection pipeline found {len(subscriptions)} subscriptions and {len(verified_utilities)} utilities"
    )

    return output_data
