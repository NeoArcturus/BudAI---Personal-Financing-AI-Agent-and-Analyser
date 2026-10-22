from models.status_codes import OpenBankingStatus
import json
import pandas as pd
from typing import Any
from datetime import datetime, timedelta, timezone
from config import SessionLocal
from models.database_models import Account, Bank, Transaction
import requests
from services.api_integrator.truelayer_sync import TrueLayerSync
from services.logger_setup import get_core_logger, log_pipeline_data

logger = get_core_logger(__name__)

class AccountReader:
    def __init__(self, user_id=None):
        self.user_id = user_id

    def get_all_accounts(self):
        all_accounts = []
        provider_logos = {}
        provider_names = {}
        try:
            providers_res = requests.get(
                "https://auth.truelayer.com/api/providers", timeout=10)
            if providers_res is not None and providers_res.status_code == 200:
                for p in providers_res.json():
                    provider_logos[p.get("provider_id")] = p.get("logo_url")
                    provider_names[p.get("provider_id")] = p.get(
                        "display_name")
        except Exception as e:
            logger.error(f"Error fetching TrueLayer providers: {e}", exc_info=True)

        try:
            with SessionLocal() as session:
                updated_banks = session.query(Bank).filter_by(
                    user_uuid=self.user_id).all()
                for b in updated_banks:
                    logo_url = provider_logos.get(b.truelayer_provider_id, "")
                    display_name = provider_names.get(
                        b.truelayer_provider_id, b.bank_name)
                    if b.consent_status == OpenBankingStatus.BANK_REVOKED_CONSENT.value:
                        all_accounts.append({
                            "account_id": b.truelayer_provider_id,
                            "bank_name": display_name,
                            "provider_name": display_name,
                            "account_number": "****",
                            "sort_code": "00-00-00",
                            "currency": "GBP",
                            "balance": 0.0,
                            "status": "revoked",
                            "consent_status": b.consent_status,
                            "bank_uuid": b.bank_uuid,
                            "provider_id": b.truelayer_provider_id,
                            "logo_url": logo_url
                        })
                        continue
                    for acc in b.accounts:
                        acc_id = str(acc.account_id)
                        last_sync = acc.last_synced_at
                        if last_sync and last_sync.tzinfo is None:
                            last_sync = last_sync.replace(tzinfo=timezone.utc)
                        now_utc = datetime.now(timezone.utc)

                        if last_sync is None or (now_utc - last_sync).total_seconds() > 300:
                            try:
                                sync = TrueLayerSync(self.user_id)
                                enc_token = b.access_token
                                if isinstance(enc_token, memoryview):
                                    enc_token = enc_token.tobytes()
                                access_token = sync.cipher_suite.decrypt(bytes(enc_token)).decode()
                                bal_res = sync._make_request(f"{sync.base_url}/{acc_id}/balance", access_token, b.truelayer_provider_id)
                                if bal_res:
                                    if bal_res.status_code == 200:
                                        bal_data = bal_res.json().get("results", [{}])[0]
                                        acc.account_balance = float(bal_data.get("available", bal_data.get("current", 0.0)))
                                        acc.last_synced_at = now_utc
                                        session.commit()
                            except Exception as e:
                                session.rollback()
                                logger.error(f"Failed direct balance sync for {acc_id}: {e}")

                        all_accounts.append({
                            "account_id": acc.account_id,
                            "bank_name": display_name,
                            "provider_name": display_name,
                            "account_number": acc.account_number,
                            "sort_code": acc.sort_code or "",
                            "currency": acc.currency or "GBP",
                            "balance": acc.account_balance or 0.0,
                            "status": "active",
                            "consent_status": b.consent_status,
                            "bank_uuid": b.bank_uuid,
                            "provider_id": b.truelayer_provider_id,
                            "logo_url": logo_url
                        })

            log_pipeline_data(
                logger=logger,
                pipeline="account_reader",
                stage="get_all_accounts",
                data={
                    "user_id": self.user_id,
                    "accounts_count": len(all_accounts),
                    "accounts": [
                        {
                            "account_id": a["account_id"],
                            "bank_name": a["bank_name"],
                            "balance": a["balance"],
                            "currency": a["currency"],
                            "status": a["status"]
                        }
                        for a in all_accounts
                    ]
                },
                message=f"Fetched {len(all_accounts)} accounts for user {self.user_id}"
            )
            return all_accounts
        except Exception:
            logger.error("Error retrieving accounts", exc_info=True)
            return []

    def get_account_balance(self, bank_name_or_id, user_uuid, account_type="TRANSACTION"):
        with SessionLocal() as session:
            acc = session.query(Account).join(Bank).filter(
                (Bank.bank_name.ilike(f"%{bank_name_or_id}%")) | (
                    Account.account_id == bank_name_or_id),
                Account.user_uuid == user_uuid
            ).first()
        if acc and acc.account_balance is not None:
            return float(acc.account_balance)
        return 0.0

    def get_transactions(self, account_id: str, user_uuid, start_date=None, end_date=None, expense_only=False):
        try:
            from models.database_models import MerchantKnowledge
            with SessionLocal() as session:
                query = session.query(Transaction, MerchantKnowledge).outerjoin(
                    MerchantKnowledge, Transaction.merchant_knowledge_uuid == MerchantKnowledge.knowledge_uuid
                ).filter(Transaction.account_id == account_id, Transaction.user_uuid == user_uuid)

                if start_date:
                    query = query.filter(Transaction.date >= start_date)
                if end_date:
                    query = query.filter(Transaction.date <= end_date)
                if expense_only:
                    query = query.filter(Transaction.amount < 0)
                
                results = query.order_by(Transaction.date.desc()).all()
                if not results:
                    log_pipeline_data(
                        logger=logger,
                        pipeline="account_reader",
                        stage="get_transactions",
                        data={
                            "account_id": account_id,
                            "user_uuid": user_uuid,
                            "count": 0,
                            "start_date": str(start_date) if start_date else None,
                            "end_date": str(end_date) if end_date else None,
                        },
                        message=f"No transactions found for account {account_id}"
                    )
                    return pd.DataFrame()
                
                transactions = []
                for tx, mk in results:
                    transactions.append({
                        "transaction_id": tx.transaction_uuid,
                        "timestamp": tx.date.isoformat() if tx.date else None,
                        "date": tx.date.isoformat() if tx.date else None,
                        "amount": tx.amount,
                        "currency": tx.currency,
                        "description": tx.description,
                        "category": mk.category if mk else "Uncategorized",
                        "sub_category": mk.sub_category if mk else None,
                        "tags": mk.tags if mk else [],
                        "bank_uuid": tx.bank_uuid,
                        "account_id": tx.account_id
                    })

                log_pipeline_data(
                    logger=logger,
                    pipeline="account_reader",
                    stage="get_transactions",
                    data={
                        "account_id": account_id,
                        "user_uuid": user_uuid,
                        "count": len(transactions),
                        "start_date": str(start_date) if start_date else None,
                        "end_date": str(end_date) if end_date else None,
                        "date_range": {
                            "min_date": transactions[-1]["date"] if transactions else None,
                            "max_date": transactions[0]["date"] if transactions else None,
                        },
                        "sample": [
                            {
                                "date": t["date"],
                                "amount": t["amount"],
                                "desc": t["description"][:30] if t["description"] else "",
                                "category": t["category"]
                            }
                            for t in transactions[:3]
                        ]
                    },
                    message=f"Retrieved {len(transactions)} transactions for account {account_id}"
                )
                return pd.DataFrame(transactions)
        except Exception:
            logger.error("Error in get_transactions", exc_info=True)
            return pd.DataFrame()

    def get_transactions_by_account(self, account_id):
        try:
            from models.database_models import MerchantKnowledge
            with SessionLocal() as session:
                query_results = session.query(Transaction, MerchantKnowledge).outerjoin(
                    MerchantKnowledge, Transaction.merchant_knowledge_uuid == MerchantKnowledge.knowledge_uuid
                ).filter(
                    Transaction.account_id == account_id, 
                    Transaction.user_uuid == self.user_id
                ).order_by(Transaction.date.desc()).all()
                
                if not query_results:
                    return []
                results = []
                for tx, mk in query_results:
                    results.append({
                        "transaction_id": tx.transaction_uuid,
                        "timestamp": tx.date.isoformat() if tx.date else None,
                        "amount": tx.amount,
                        "description": tx.description,
                        "category": mk.category if mk else "Uncategorized",
                        "sub_category": mk.sub_category if mk else None,
                        "tags": mk.tags if mk else []
                    })

                log_pipeline_data(
                    logger=logger,
                    pipeline="account_reader",
                    stage="get_transactions_by_account",
                    data={
                        "account_id": account_id,
                        "user_id": self.user_id,
                        "count": len(results),
                        "sample": [
                            {
                                "amount": r["amount"],
                                "desc": r["description"][:30] if r["description"] else "",
                                "category": r["category"]
                            }
                            for r in results[:3]
                        ]
                    },
                    message=f"Retrieved {len(results)} transactions by account {account_id}"
                )
                return results
        except Exception:
            logger.error("An error occurred in get_transactions_by_account", exc_info=True)
            return []
