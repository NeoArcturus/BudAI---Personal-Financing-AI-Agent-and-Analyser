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
        batch_size = 10 # Matches Qwen-4B limits
        semaphore = asyncio.Semaphore(1)
        
        async def sem_task(batch):
            async with semaphore:
                res = await agent._categorize_batch(batch)
                await asyncio.sleep(2.0) # rate limiting
                return res
        
        batches = [transactions[i:i + batch_size] for i in range(0, len(transactions), batch_size)]
        
        # User requested tqdm progress bar
        results = []
        for batch in tqdm(batches, desc="Categorizing Transactions", unit="batch"):
            res = await sem_task(batch)
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
        
    except Exception as e:
        logger.error(json.dumps({"message": f"Error in categorize_specific_transactions_bg: {e}", "status_code": 500}), exc_info=True)
