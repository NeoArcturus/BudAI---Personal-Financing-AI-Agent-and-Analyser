import pandas as pd
from typing import Any
from config import SessionLocal
from models.database_models import Account, Bank, Transaction
import requests
from services.logger_setup import get_core_logger

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
                "https://auth.truelayer.com/api/providers")
            if providers_res is not None and providers_res.status_code == 200:
                for p in providers_res.json():
                    provider_logos[p.get("provider_id")] = p.get("logo_url")
                    provider_names[p.get("provider_id")] = p.get(
                        "display_name")
        except Exception as e:
            logger.error("Error fetching providers", exc_info=True)

        try:
            with SessionLocal() as session:
                updated_banks = session.query(Bank).filter_by(
                    user_uuid=self.user_id).all()
                for b in updated_banks:
                    logo_url = provider_logos.get(b.truelayer_provider_id, "")
                    display_name = provider_names.get(
                        b.truelayer_provider_id, b.bank_name)
                    if b.consent_status == 'revoked':
                        all_accounts.append({
                            "account_id": b.truelayer_provider_id,
                            "bank_name": display_name,
                            "provider_name": display_name,
                            "account_number": "****",
                            "sort_code": "00-00-00",
                            "currency": "GBP",
                            "balance": 0.0,
                            "status": "revoked",
                            "provider_id": b.truelayer_provider_id,
                            "logo_url": logo_url
                        })
                        continue
                    for acc in b.accounts:
                        all_accounts.append({
                            "account_id": acc.account_id,
                            "bank_name": display_name,
                            "provider_name": display_name,
                            "account_number": acc.account_number,
                            "sort_code": acc.sort_code or "",
                            "currency": acc.currency or "GBP",
                            "balance": acc.account_balance or 0.0,
                            "status": "active",
                            "provider_id": b.truelayer_provider_id,
                            "logo_url": logo_url
                        })
            return all_accounts
        except Exception:
            logger.error("Error getting all accounts", exc_info=True)
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

    def get_transactions(self, identifier, user_uuid, start_date=None, end_date=None, expense_only=False):
        try:
            with SessionLocal() as session:
                query = session.query(Transaction)
                identifiers = [identifier] if isinstance(identifier, str) else identifier
                if identifiers and "ALL" not in [str(i).upper() for i in identifiers]:
                    query = query.join(Account).join(Bank).filter(
                        (Bank.bank_name.in_(identifiers)) | (Account.account_id.in_(identifiers))
                    )
                query = query.filter(Transaction.user_uuid == user_uuid)

                if start_date:
                    query = query.filter(Transaction.date >= start_date)
                if end_date:
                    query = query.filter(Transaction.date <= end_date)
                if expense_only:
                    query = query.filter(Transaction.amount < 0)
                
                txs = query.order_by(Transaction.date.desc()).all()
                if not txs:
                    return pd.DataFrame()
                
                transactions = []
                for tx in txs:
                    transactions.append({
                        "transaction_id": tx.transaction_uuid,
                        "timestamp": tx.date.isoformat() if tx.date else None,
                        "date": tx.date.isoformat() if tx.date else None,
                        "amount": tx.amount,
                        "currency": tx.currency,
                        "description": tx.description,
                        "category": tx.category,
                        "bank_uuid": tx.bank_uuid,
                        "account_id": tx.account_id
                    })
                return pd.DataFrame(transactions)
        except Exception:
            logger.error("Error in get_transactions", exc_info=True)
            return pd.DataFrame()

    def get_transactions_by_account(self, account_id):
        try:
            with SessionLocal() as session:
                txs = session.query(Transaction).filter_by(
                    account_id=account_id, user_uuid=self.user_id).order_by(Transaction.date.desc()).all()
                if not txs:
                    return []
                results = []
                for tx in txs:
                    results.append({
                        "transaction_id": tx.transaction_uuid,
                        "timestamp": tx.date.isoformat() if tx.date else None,
                        "amount": tx.amount,
                        "description": tx.description,
                        "category": tx.category
                    })
                return results
        except Exception:
            logger.error("An error occurred in get_transactions_by_account", exc_info=True)
            return []
