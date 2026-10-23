import config
from typing import Dict, Any
from prefect import flow, task, get_run_logger
from services.analytics.anchor_service import get_anchor_service

@task(
    name="reconcile_merchant_anchors_task",
    retries=2,
    retry_delay_seconds=30,
    tags=["vector-reconciliation"]
)
def reconcile_merchant_anchors_task(limit: int = 1000) -> Dict[str, Any]:
    """
    Prefect task that scans for transactions with merchant_knowledge_uuid IS NULL,
    computes embeddings via FastEmbed, clusters them into Merchant Anchors,
    and updates both PostgreSQL and Qdrant.
    """
    logger = get_run_logger()
    logger.info("Scanning for unlinked/unvectorized transactions...")
    anchor_service = get_anchor_service()
    result = anchor_service.reconcile_unlinked_transactions(limit=limit)

    status = result.get("status")
    if status == "noop":
        logger.info("Zero unlinked transactions found. System is fully reconciled.")
    else:
        logger.info(
            f"Successfully processed {result.get('transactions_linked')} transactions "
            f"into {result.get('anchors_created')} distinct merchant anchors."
        )
    return result

@flow(
    name="Merchant Vectorization and Reconciliation Flow",
    description="Automated cron/event flow to vectorize unlinked transactions and cluster them into Merchant Anchors"
)
def reconcile_and_vectorize_merchants_flow(limit: int = 1000) -> Dict[str, Any]:
    return reconcile_merchant_anchors_task(limit=limit)

if __name__ == "__main__":
    reconcile_and_vectorize_merchants_flow()
