from prefect import serve
from prefect.events import DeploymentEventTrigger
from services.orchestration.periodic_worker import run_periodic_evaluation
from services.orchestration.cron_flows import (
    run_global_subscription_analytics_flow,
    run_global_hdbscan_clustering_flow,
    train_global_model_flow,
    oss_timescale_refresh_flow,
    expire_stale_subscriptions_flow,
    oss_vector_optimizer_flow,
    oss_monte_carlo_sim_flow,
    oss_token_auditor_flow,
    llm_categorization_sweep_flow,
    oss_database_reconciliation_flow
)

if __name__ == "__main__":
    subscription_analytics = run_global_subscription_analytics_flow.to_deployment(
        name="oss-subscription-analytics",
        cron="*/10 * * * *",
        triggers=[
            DeploymentEventTrigger(
                expect={"prefect.flow-run.Completed"},
                match_related={"prefect.resource.name": "cloud-onboarding"},
            )
        ]
    )
    
    hdbscan_clustering = run_global_hdbscan_clustering_flow.to_deployment(
        name="oss-hdbscan-clustering",
        cron="*/10 * * * *",
        triggers=[
            DeploymentEventTrigger(
                expect={"prefect.flow-run.Completed"},
                match_related={"prefect.resource.name": "cloud-onboarding"},
            )
        ]
    )
    
    model_training = train_global_model_flow.to_deployment(
        name="oss-model-training",
        triggers=[
            DeploymentEventTrigger(
                expect={"prefect.flow-run.Completed"},
                match_related={"prefect.resource.name": "cloud-failsafe-sync"},
            )
        ]
    )

    timescale_refresh = oss_timescale_refresh_flow.to_deployment(
        name="oss-timescale-refresh", 
        cron="*/15 * * * *"
    )
    
    subscription_sweeper = expire_stale_subscriptions_flow.to_deployment(
        name="oss-subscription-sweeper", 
        cron="0 2 * * *"
    )
    
    vector_optimizer = oss_vector_optimizer_flow.to_deployment(
        name="oss-vector-optimizer", 
        cron="0 2 * * 0" # Weekly on Sunday
    )
    
    monte_carlo = oss_monte_carlo_sim_flow.to_deployment(
        name="oss-monte-carlo-sim", 
        cron="0 3 * * *"
    )
    
    token_auditor = oss_token_auditor_flow.to_deployment(
        name="oss-token-auditor", 
        cron="0 4 * * *"
    )
    
    categorization_sweeper = llm_categorization_sweep_flow.to_deployment(
        name="oss-llm-categorization-sweeper",
        cron="*/10 * * * *"
    )
    
    db_reconciliation = oss_database_reconciliation_flow.to_deployment(
        name="oss-database-reconciliation",
        cron="0 1 * * *" # Nightly at 1 AM
    )
    
    periodic_evaluator = run_periodic_evaluation.to_deployment(
        name="Periodic-Evaluator",
        cron="*/5 * * * *",
        tags=["ai-evaluation", "system-core"]
    )
    
    print("Serving Open Source Deployments...")
    serve(
        subscription_analytics, 
        hdbscan_clustering,
        model_training, 
        timescale_refresh, 
        subscription_sweeper, 
        vector_optimizer, 
        monte_carlo, 
        token_auditor,
        categorization_sweeper,
        db_reconciliation,
        periodic_evaluator
    )
