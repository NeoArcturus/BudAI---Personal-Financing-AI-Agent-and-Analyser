from langfuse.langchain import CallbackHandler

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
    from agents.core_financial.Analyser_Agent.AnalyserAgent import AnalyserAgent
    from langchain_core.messages import SystemMessage, HumanMessage
    from config import SessionLocal
    from models.database_models import User
    
    agent = AnalyserAgent()
    
    with SessionLocal() as session:
        user_uuids = [u[0] for u in session.query(User.user_uuid).all()]
        
    for user_id in user_uuids:
        messages = [
            SystemMessage(content="You are the Analyser Agent. Review the user's spending trends and decide whether to send a system alert. Use tools."),
            HumanMessage(content=f"Please analyze spending trends for user {user_id}.")
        ]
        try:
            agent.app.invoke({"messages": messages, "user_id": user_id}, config={"callbacks": [CallbackHandler()], "metadata": {"langfuse_session_id": user_id, "langfuse_tags": ["proactive-insights"]}})
        except Exception as e:
            from services.logger_setup import get_core_logger
            logger = get_core_logger(__name__)
            logger.error(f"Failed to generate proactive insights for {user_id}: {e}")

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
        from agents.core_financial.Categorizer_Agent.CategorizerAgent import CategorizerAgent
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
    try:
        from agents.infrastructure.CloudFailsafe_Agent.CloudFailsafeAgent import CloudFailsafeAgent
        from langchain_core.messages import HumanMessage
        agent = CloudFailsafeAgent()
        # Use SYSTEM as user_uuid to sweep all missing webhooks
        agent.app.invoke({"messages": [HumanMessage(content="Trigger failsafe sync for all missing webhook data.")], "user_uuid": "SYSTEM"}, config={"callbacks": [CallbackHandler()], "metadata": {"langfuse_session_id": "SYSTEM", "langfuse_tags": ["failsafe-sync"]}})
    except Exception as e:
        logger.error(f"Failsafe sync failed: {e}")

@flow(name="Timescale Aggregate Refresh")
def oss_timescale_refresh_flow():
    logger = get_run_logger()
    logger.info("Refreshing continuous aggregates...")
    try:
        from config import engine
        from sqlalchemy import text
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            # conn.execute(text("CALL refresh_continuous_aggregate('rolling_user_stats', NULL, NULL);"))  # Temporarily disabled: view does not exist
    except Exception as e:
        logger.error(f"Failed to refresh aggregates: {e}")

@flow(name="Vector Optimizer")
def oss_vector_optimizer_flow():
    logger = get_run_logger()
    logger.info("Optimizing pgvector indexes...")
    from agents.infrastructure.VectorMaintenance_Agent.VectorMaintenanceAgent import VectorMaintenanceAgent
    agent = VectorMaintenanceAgent()
    agent.optimize_rag_memory()

@flow(name="Database Reconciliation")
def oss_database_reconciliation_flow():
    logger = get_run_logger()
    logger.info("Auditing zero-sum math...")
    from agents.infrastructure.Reconciliation_Agent.ReconciliationAgent import DatabaseReconciliationAgent
    agent = DatabaseReconciliationAgent()
    agent.audit_zero_sum_math()

@flow(name="Monte Carlo Simulator")
def oss_monte_carlo_sim_flow():
    logger = get_run_logger()
    logger.info("Running Monte Carlo simulations...")
    try:
        from agents.intelligence.MonteCarloSimulation_Agent.MonteCarloSimulationAgent import MonteCarloSimulationAgent
        from langchain_core.messages import HumanMessage
        agent = MonteCarloSimulationAgent()
        agent.app.invoke({"messages": [HumanMessage(content="Run 1000 Monte Carlo simulations to stress test runway.")], "user_uuid": "SYSTEM"}, config={"callbacks": [CallbackHandler()], "metadata": {"langfuse_session_id": "SYSTEM", "langfuse_tags": ["monte-carlo"]}})
    except Exception as e:
        logger.error(f"Monte Carlo sim failed: {e}")

@flow(name="Token Auditor")
def oss_token_auditor_flow():
    logger = get_run_logger()
    logger.info("Auditing LLM token usage...")
    try:
        from agents.infrastructure.TokenAuditing_Agent.TokenAuditingAgent import TokenAuditingAgent
        from langchain_core.messages import HumanMessage
        agent = TokenAuditingAgent()
        agent.app.invoke({"messages": [HumanMessage(content="Audit system-wide LLM token usage.")] , "user_uuid": "SYSTEM"}, config={"callbacks": [CallbackHandler()], "metadata": {"langfuse_session_id": "SYSTEM", "langfuse_tags": ["token-audit"]}})
    except Exception as e:
        logger.error(f"Token audit failed: {e}")

@task(retries=1, tags=["alienware-llm"])
def llm_categorization_sweep_task():
    logger = get_run_logger()
    logger.info("Starting Agentic Categorization Sweep...")
    
    from sqlalchemy import text
    from config import SessionLocal
    from agents.core_financial.Categorizer_Agent.CategorizerAgent import CategorizerAgent
    from langchain_core.messages import SystemMessage, HumanMessage
    
    agent = CategorizerAgent()
    
    with SessionLocal() as session:
        # Get orphans missing their Semantic Foreign Key
        query = text("SELECT transaction_uuid, semi_cleaned_description FROM transactions WHERE merchant_knowledge_uuid IS NULL LIMIT 50")
        results = session.execute(query).fetchall()
        
        if not results:
            logger.info("No uncategorized transactions found.")
            return
            
        
        batch_data = [{"transaction_uuid": r[0], "merchant_name": r[1]} for r in results]
        import json
        
        logger.info(f"Delegating categorization of {len(batch_data)} transactions to CategorizerAgent in a single batch...")
        messages = [
            HumanMessage(content=f"Please categorize the following batch of {len(batch_data)} transactions:\n{json.dumps(batch_data, indent=2)}")
        ]
        
        agent.app.invoke(
            {"messages": messages, "transaction_uuid": "batch", "merchant_name": "batch"}, 
            config={"callbacks": [CallbackHandler()], "metadata": {"langfuse_session_id": "SYSTEM", "langfuse_tags": ["categorization-sweep"]}}
        )
            
        logger.info("Agentic categorization sweep complete.")

@flow(name="LLM Categorization Sweeper")
def llm_categorization_sweep_flow():
    llm_categorization_sweep_task()

@task(retries=1, retry_delay_seconds=60)
def autonomous_fiduciary_patrol_task():
    logger = get_run_logger()
    logger.info("Executing 10-Minute Autonomous Fiduciary Patrol...")
    try:
        from services.orchestrator_graph import get_orchestrator_app
        from langchain_core.messages import HumanMessage
        from config import SessionLocal
        from models.database_models import User
        
        app = get_orchestrator_app()
        
        with SessionLocal() as session:
            user_uuids = [u[0] for u in session.query(User.user_uuid).all()]
            
        for user_id in user_uuids:
            from langfuse.langchain import CallbackHandler
            import time
            
            logger.info(f"Patrolling user {user_id}")
            
            langfuse_handler = CallbackHandler()
            app_config = {"callbacks": [langfuse_handler], "metadata": {"langfuse_session_id": f"patrol_{user_id}_{int(time.time())}", "langfuse_user_id": str(user_id), "langfuse_tags": ["10-minute-patrol", "autonomous"]}}
            
            # This triggers the Langfuse-backed system prompt to evaluate Liability Horizon and Pending TXs
            app.invoke({
                "messages": [HumanMessage(content="Wake up. Execute the 10-minute patrol. Fetch the Liability Horizon, check Pending Transactions, and perform any necessary Virtual PIS Sweeps to protect Tier 1 and Tier 2 goals.")],
                "user_uuid": user_id
            }, {"configurable": {"thread_id": f"patrol_{user_id}"}, **app_config})
            
    except Exception as e:
        logger.error(f"Fiduciary Patrol failed: {e}")

@flow(name="10-Minute Autonomous Patroller")
def autonomous_fiduciary_patrol_flow():
    autonomous_fiduciary_patrol_task()
