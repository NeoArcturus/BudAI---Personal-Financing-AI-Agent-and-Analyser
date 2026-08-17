import time
import json
import asyncio
from tqdm import tqdm
from sqlalchemy import text
from config import SessionLocal
from services.Categorizer_Agent.CategorizerAgent import CategorizerAgent
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

async def categorize_specific_transactions_bg(tx_uuids, user_uuid):
    if not tx_uuids:
        return
        
    try:
        from models.database_models import Transaction
        with SessionLocal() as session:
            txs = session.query(Transaction).filter(
                Transaction.transaction_uuid.in_(tx_uuids),
                Transaction.user_uuid == str(user_uuid),
                (Transaction.category == 'Uncategorized') | (Transaction.category == None) | (Transaction.category == '')
            ).all()
            
            if not txs:
                logger.info(json.dumps({"message": f"No uncategorized transactions found for the given UUIDs.", "status_code": 200}))
                return
                
            transactions = []
            for tx in txs:
                transactions.append({
                    "transaction_uuid": tx.transaction_uuid,
                    "date": str(tx.date.date()) if tx.date else "",
                    "raw_string": tx.description or '',
                    "semi_cleaned_string": tx.semi_cleaned_description or '',
                    "fully_cleaned_string": tx.fully_cleaned_description or '',
                    "amount": float(tx.amount) if tx.amount else 0.0,
                })
                
        logger.info(json.dumps({"message": f"Lazy ML Categorization started for {len(transactions)} transactions.", "status_code": 200}))
        
        agent = CategorizerAgent()
        batch_size = 50
        semaphore = asyncio.Semaphore(3)
        
        batches = [transactions[i:i + batch_size] for i in range(0, len(transactions), batch_size)]
        total_batches = len(batches)
        
        async def sem_task(batch_idx, batch):
            async with semaphore:
                start_time = time.perf_counter()
                logger.info(json.dumps({
                    "message": f"Batch {batch_idx + 1}/{total_batches} started ({len(batch)} transactions)",
                    "batch_id": batch_idx + 1,
                    "status_code": 100
                }))
                
                res = await agent._categorize_batch(batch)
                
                elapsed = time.perf_counter() - start_time
                logger.info(json.dumps({
                    "message": f"Batch {batch_idx + 1}/{total_batches} finished in {elapsed:.2f}s",
                    "batch_id": batch_idx + 1,
                    "status_code": 200
                }))
                return res
        
        tasks = [sem_task(idx, batch) for idx, batch in enumerate(batches)]
        results = []
        
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Categorizing Transactions", unit="batch"):
            res = await coro
            results.append(res)
            
        categorized_list = [item for sublist in results for item in sublist]
        
        with SessionLocal() as session:
            updated_count = 0
            for item in categorized_list:
                existing = session.query(Transaction).filter_by(transaction_uuid=item.transaction_uuid).first()
                if existing:
                    existing.category = item.category
                    existing.sub_category = item.sub_category
                    updated_count += 1
            session.commit()
            
        logger.info(json.dumps({"message": f"Lazy ML Categorization completed. {updated_count} transactions categorized and saved.", "status_code": 200}))
        
        from utils.state_manager import clear_account_state
        account_ids = list(set(tx.account_id for tx in txs if tx.account_id))
        for acc_id in account_ids:
            clear_account_state(acc_id)
            
    except Exception as e:
        logger.error(json.dumps({"message": f"Error in categorize_specific_transactions_bg: {e}", "status_code": 500}), exc_info=True)
        if 'txs' in locals():
            from utils.state_manager import clear_account_state
            for tx in txs:
                if tx.account_id:
                    clear_account_state(tx.account_id)