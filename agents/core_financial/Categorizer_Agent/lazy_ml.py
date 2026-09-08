import time
import json
import asyncio
from tqdm import tqdm
from sqlalchemy import text
from config import SessionLocal
from agents.core_financial.Categorizer_Agent.CategorizerAgent import CategorizerAgent
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

async def categorize_specific_transactions_bg(tx_uuids, user_uuid):
    if not tx_uuids:
        return
        
    try:
        from models.database_models import Transaction
        with SessionLocal() as session:
            from models.database_models import MerchantKnowledge
            txs_query = session.query(Transaction).outerjoin(
                MerchantKnowledge, Transaction.merchant_knowledge_uuid == MerchantKnowledge.knowledge_uuid
            ).filter(
                Transaction.transaction_uuid.in_(tx_uuids),
                Transaction.user_uuid == str(user_uuid),
                (MerchantKnowledge.category == 'Uncategorized') | 
                (MerchantKnowledge.category == None) | 
                (MerchantKnowledge.category == '') | 
                (MerchantKnowledge.sub_category == None) | 
                (MerchantKnowledge.tags == None) | 
                (text("merchant_knowledge.tags::text = '[]'"))
            )
            txs = txs_query.all()
            
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
        semaphore = asyncio.Semaphore(5)
        
        async def process_transaction(tx, sem):
            from langchain_core.messages import HumanMessage
            async with sem:
                state = {
                    "messages": [HumanMessage(content=f"Categorize this transaction. Description: '{tx['raw_string']}', Amount: {tx['amount']}, Date: {tx['date']}")],
                    "user_uuid": str(user_uuid),
                    "transaction_uuid": tx["transaction_uuid"],
                    "merchant_name": tx["semi_cleaned_string"]
                }
                try:
                    await agent.app.ainvoke(state)
                    return True
                except Exception as e:
                    logger.error(json.dumps({"message": f"Failed to categorize tx {tx['transaction_uuid']}: {e}", "status_code": 500}))
                    return False
        
        tasks = [process_transaction(tx, semaphore) for tx in transactions]
        
        updated_count = 0
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Categorizing Transactions"):
            if await coro:
                updated_count += 1
            
        logger.info(json.dumps({"message": f"Lazy ML Categorization completed. {updated_count} transactions processed.", "status_code": 200}))
        
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