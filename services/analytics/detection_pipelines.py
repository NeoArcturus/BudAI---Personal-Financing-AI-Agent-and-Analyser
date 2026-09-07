import json
import os
from sqlmodel import Session
from sqlalchemy import text
from services.logger_setup import get_core_logger
from langchain_openai import OpenAIEmbeddings

logger = get_core_logger(__name__)

def run_detection_pipelines(session: Session, user_uuid: str) -> dict:
    """
    Step 2 of Phase 3: Mathematical & Vector Detection
    Runs SQL window functions and pgvector math to find Subscriptions and Utilities.
    """
    logger.info(json.dumps({"message": f"Running detection pipelines for {user_uuid}", "status_code": 200}))
    
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
            
    # 3. Vector Verification for Utilities
    verified_utilities = []
    if potential_utilities:
        try:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
            if not base_url.endswith("/v1"): 
                base_url = f"{base_url}/v1"
                
            embeddings_model = OpenAIEmbeddings(
                base_url=base_url,
                model="text-embedding-nomic-embed-text-v1.5",
                api_key="budai-local",
                check_embedding_ctx_length=False
            )
            
            merchants_to_embed = [u["merchant"] for u in potential_utilities if u["merchant"]]
            if merchants_to_embed:
                vectors = embeddings_model.embed_documents(merchants_to_embed)
                
                for idx, u in enumerate(potential_utilities):
                    if not u["merchant"]: continue
                    v = vectors[idx]
                    
                    vector_query = text("""
                        SELECT category, clean_merchant_name, (embedding <=> CAST(:vec AS vector)) as distance
                        FROM merchant_knowledge 
                        WHERE category = 'Utility'
                        ORDER BY embedding <=> CAST(:vec AS vector)
                        LIMIT 1
                    """)
                    result_dist = session.execute(vector_query, {"vec": str(v)}).first()
                    
                    # If semantic match is tight (< 0.1), it's a verified utility
                    if result_dist and result_dist.distance < 0.1:
                        verified_utilities.append(u)
                        
        except Exception as e:
            logger.error(json.dumps({"message": f"Vector search failed in detection pipeline: {e}", "status_code": 500}))
            
    return {
        "new_subscriptions": subscriptions,
        "new_utilities": verified_utilities
    }
