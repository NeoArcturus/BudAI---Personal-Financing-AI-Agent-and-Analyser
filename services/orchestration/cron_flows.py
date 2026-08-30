from prefect import flow, task, get_run_logger
from config import SessionLocal
from models.database_models import Bank, User

@task(retries=3, retry_delay_seconds=60)
def refresh_tokens_task():
    logger = get_run_logger()
    from services.api_integrator.access_token_generator import AccessTokenGenerator
    try:
        with SessionLocal() as session:
            providers = session.query(Bank.bank_name, Bank.truelayer_provider_id, Bank.user_uuid).all()
        if not providers:
            return
        token_gen = AccessTokenGenerator()
        for bank_name, provider_id, user_uuid in providers:
            try:
                success = token_gen.refresh_token(provider_id, user_uuid)
                if success:
                    logger.debug(f"Successfully refreshed tokens for {bank_name}.")
                else:
                    logger.warning(f"Failed to refresh tokens for {bank_name}.")
            except Exception as e:
                logger.error(f"Error refreshing {bank_name}: {e}")
    except Exception as e:
        logger.error(f"Critical error in token refresh task: {e}")
        raise e

@flow(name="Bank Token Refresh")
def refresh_all_tokens_flow():
    refresh_tokens_task()

@task(retries=2, retry_delay_seconds=30, tags=["alienware-llm"])
def analyze_subscriptions_task():
    logger = get_run_logger()
    from services.analytics.subscription_detector import SubscriptionDetector
    logger.info("Starting global background job: Subscription Analytics")
    
    with SessionLocal() as session:
        user_uuids = [u[0] for u in session.query(User.user_uuid).all()]
        
    if not user_uuids:
        return
        
    sub_detector = SubscriptionDetector()
    for user_uuid in user_uuids:
        try:
            sub_detector.analyze_user_subscriptions(user_uuid)
        except Exception as e:
            logger.error(f"Failed subscription analytics for user {user_uuid}: {e}")
            
    logger.info("Global background job completed: Subscription Analytics")

@flow(name="Subscription Analytics")
def run_global_subscription_analytics_flow():
    analyze_subscriptions_task()

@task(retries=2, retry_delay_seconds=30, tags=["alienware-llm"])
def analyze_hdbscan_clustering_task():
    logger = get_run_logger()
    from services.analytics.lifestyle_clustering import LifestyleClusteringService
    logger.info("Starting global background job: HDBSCAN Clustering")
    
    with SessionLocal() as session:
        user_uuids = [u[0] for u in session.query(User.user_uuid).all()]
        
    if not user_uuids:
        return
        
    cluster_service = LifestyleClusteringService()
    for user_uuid in user_uuids:
        try:
            cluster_service.analyze_user_lifestyle(user_uuid)
        except Exception as e:
            logger.error(f"Failed HDBSCAN clustering for user {user_uuid}: {e}")
            
    logger.info("Global background job completed: HDBSCAN Clustering")

@flow(name="HDBSCAN Lifestyle Clustering")
def run_global_hdbscan_clustering_flow():
    analyze_hdbscan_clustering_task()

@task(retries=2, retry_delay_seconds=120)
def generate_proactive_insights_task():
    from services.analytics.proactive_insights import generate_proactive_insights_for_all_users
    generate_proactive_insights_for_all_users()

@flow(name="Nightly Proactive Insights")
def generate_proactive_insights_flow():
    generate_proactive_insights_task()

@task(retries=1, retry_delay_seconds=30)
def expire_stale_subscriptions_task():
    from services.analytics.subscription_sweeper import expire_stale_subscriptions
    expire_stale_subscriptions()

@flow(name="Subscription Sweeper")
def expire_stale_subscriptions_flow():
    expire_stale_subscriptions_task()

@task(retries=3, retry_delay_seconds=300, tags=["alienware-llm"])
def train_global_model_task():
    logger = get_run_logger()
    try:
        from services.memory_service import MemoryService
        logger.info("Pre-warming local ML embedding model...")
        MemoryService()
        logger.info("ML embedding model initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize MemoryService: {e}")
        raise e
        
    try:
        from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
        agent = CategorizerAgent()
        res = agent.train_global()
        if res.get("trained"):
            logger.debug(f"Global categorization model trained on {res.get('samples', 0)} samples.")
        else:
            logger.debug(f"Global categorization model initialization: {res.get('reason')}")
    except Exception as e:
        logger.error(f"Failed to initialize global categorizer: {e}")
        raise e

@flow(name="Global Model Training and ML Init")
def train_global_model_flow():
    train_global_model_task()

@flow(name="Nightly Maintenance Routine")
def nightly_maintenance_flow():
    # Consolidating these to stay under the Prefect Free Tier deployment limit
    analyze_subscriptions_task()
    analyze_hdbscan_clustering_task()
    expire_stale_subscriptions_task()
    generate_proactive_insights_task()

@flow(name="Daily TrueLayer Failsafe")
def cloud_failsafe_sync_flow():
    logger = get_run_logger()
    logger.info("Executing daily catch-all TrueLayer sync...")
    # TODO: Implement catch-all sync

@flow(name="Timescale Aggregate Refresh")
def oss_timescale_refresh_flow():
    logger = get_run_logger()
    logger.info("Refreshing continuous aggregates...")
    try:
        from config import engine
        from sqlalchemy import text
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("CALL refresh_continuous_aggregate('rolling_user_stats', NULL, NULL);"))
    except Exception as e:
        logger.error(f"Failed to refresh aggregates: {e}")

@flow(name="Vector Optimizer")
def oss_vector_optimizer_flow():
    logger = get_run_logger()
    logger.info("Optimizing pgvector indexes...")
    # TODO: Implement VACUUM ANALYZE for vector tables

@flow(name="Monte Carlo Simulator")
def oss_monte_carlo_sim_flow():
    logger = get_run_logger()
    logger.info("Running Monte Carlo simulations...")
    # TODO: Implement stress tests

@flow(name="Token Auditor")
def oss_token_auditor_flow():
    logger = get_run_logger()
    logger.info("Auditing LLM token usage...")
    # TODO: Implement token audit

@task(retries=2, retry_delay_seconds=30, tags=["alienware-llm"])
def llm_categorization_sweep_task():
    logger = get_run_logger()
    logger.info("Starting LLM Categorization Sweep...")
    import os
    import json
    import uuid
    from datetime import datetime
    from sqlalchemy import text
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from config import SessionLocal
    
    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
    if not base_url.endswith("/v1"): 
        base_url = f"{base_url}/v1"
        
    chat_model = ChatOpenAI(
        base_url=base_url,
        model="qwen2.5-coder:7b",
        api_key="budai-local",
        temperature=0.0
    )
    embeddings_model = OpenAIEmbeddings(
        base_url=base_url,
        model="text-embedding-nomic-embed-text-v1.5",
        api_key="budai-local",
        check_embedding_ctx_length=False
    )
    
    with SessionLocal() as session:
        # Get Uncategorised transactions
        query = text("SELECT transaction_uuid, semi_cleaned_description FROM transactions WHERE category = 'Uncategorised' LIMIT 50")
        results = session.execute(query).fetchall()
        
        if not results:
            logger.info("No uncategorised transactions found.")
            return
            
        # Deduplicate to save LLM tokens
        tx_dict = {}
        for row in results:
            tx_id, desc = row[0], row[1]
            if desc not in tx_dict:
                tx_dict[desc] = []
            tx_dict[desc].append(tx_id)
            
        unique_merchants = list(tx_dict.keys())
        
        # Batch LLM Prompt
        prompt = f"""
You are a financial classification engine. Map the following merchants to this strict taxonomy list: 
['Food & Dining', 'Shopping & Retail', 'Transport', 'Utilities', 'Entertainment & Lifestyle', 'Healthcare', 'Subscriptions & Digital Services']. 
Return ONLY a valid JSON object matching this schema exactly: {{"classifications": [{{"merchant": "name", "category": "Category"}}]}}
Merchants: {json.dumps(unique_merchants)}
"""
        
        try:
            response = chat_model.invoke(prompt)
            data = json.loads(response.content.strip().strip("```json").strip("```"))
            classifications = data.get("classifications", [])
            
            for item in classifications:
                merchant = item.get("merchant")
                category = item.get("category")
                
                if merchant in tx_dict:
                    # Update transactions optimistically
                    tx_ids = tx_dict[merchant]
                    update_query = text("UPDATE transactions SET category = :cat WHERE transaction_uuid = ANY(:tx_ids) AND category = 'Uncategorised'")
                    session.execute(update_query, {"cat": category, "tx_ids": tx_ids})
                    
                    # Embed and Upsert into MerchantKnowledge
                    vec = embeddings_model.embed_documents([merchant])[0]
                    k_uuid = str(uuid.uuid4())
                    
                    upsert_query = text("""
                        INSERT INTO merchant_knowledge (knowledge_uuid, clean_merchant_name, category, embedding, is_human_verified, created_at)
                        VALUES (:uuid, :name, :cat, :vec, FALSE, :now)
                    """)
                    session.execute(upsert_query, {
                        "uuid": k_uuid, "name": merchant, "cat": category, 
                        "vec": str(vec), "now": datetime.utcnow()
                    })
                    
            session.commit()
            logger.info(f"Successfully swept and categorized {len(classifications)} unique merchants.")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed LLM categorization sweep: {e}")

@flow(name="LLM Categorization Sweeper")
def llm_categorization_sweep_flow():
    llm_categorization_sweep_task()

