import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from services.analytics.multicurrency_reconciliation_service import (
    MultiCurrencyReconciliationService,
    get_multicurrency_reconciliation_service
)
from models.database_models import Transaction, MerchantKnowledge


def test_singleton_accessor():
    s1 = get_multicurrency_reconciliation_service()
    s2 = get_multicurrency_reconciliation_service()
    assert s1 is s2


def test_check_user_has_multicurrency_accounts_empty():
    service = MultiCurrencyReconciliationService()
    mock_session = MagicMock()
    mock_session.execute.return_value.fetchall.return_value = []
    
    result = service.check_user_has_multicurrency_accounts(mock_session, "user-123")
    assert result == []


def test_check_user_has_multicurrency_accounts_found():
    service = MultiCurrencyReconciliationService()
    mock_session = MagicMock()
    mock_session.execute.return_value.fetchall.return_value = [("bank-abc",)]
    
    result = service.check_user_has_multicurrency_accounts(mock_session, "user-123")
    assert result == ["bank-abc"]


def test_match_and_reconcile_cluster_recharge_com():
    service = MultiCurrencyReconciliationService()
    mock_session = MagicMock()

    t0 = datetime(2026, 11, 14, 6, 36, 43, 204000, tzinfo=timezone.utc)
    
    tx_eur = {
        "transaction_uuid": "txn-eur-123",
        "account_id": "acc-eur",
        "bank_uuid": "bank-wise",
        "date": t0,
        "amount": -1.94,
        "currency": "EUR",
        "description": "Card transaction of 12.29 EUR issued by Recharge.com 794638114558 Amsterdam",
        "merchant_knowledge_uuid": None,
        "sub_category": None,
        "tags": ["#discretionary"]
    }
    tx_fee = {
        "transaction_uuid": "txn-fee-123",
        "account_id": "acc-gbp",
        "bank_uuid": "bank-wise",
        "date": t0 + timedelta(milliseconds=31),
        "amount": -0.03,
        "currency": "GBP",
        "description": "Wise Charges for: CARD-4608854962",
        "merchant_knowledge_uuid": None,
        "sub_category": "Bank Charges",
        "tags": ["#penalty", "#bank_charges"]
    }
    tx_gbp = {
        "transaction_uuid": "txn-gbp-123",
        "account_id": "acc-gbp",
        "bank_uuid": "bank-wise",
        "date": t0 + timedelta(milliseconds=32),
        "amount": -8.64,
        "currency": "GBP",
        "description": "Card transaction of 12.29 EUR issued by Recharge.com 794638114558 Amsterdam (fee: 0.03 GBP)",
        "merchant_knowledge_uuid": None,
        "sub_category": "Streaming Video & Audio",
        "tags": ["#discretionary", "#recurring"]
    }

    cluster = [tx_eur, tx_fee, tx_gbp]

    # Mock DB Transaction objects
    db_eur = MagicMock(spec=Transaction, transaction_uuid="txn-eur-123", tags=["#discretionary"], sub_category=None, merchant_knowledge_uuid=None)
    db_fee = MagicMock(spec=Transaction, transaction_uuid="txn-fee-123", tags=["#penalty", "#bank_charges"], sub_category="Bank Charges", merchant_knowledge_uuid=None)
    db_gbp = MagicMock(spec=Transaction, transaction_uuid="txn-gbp-123", tags=["#discretionary", "#recurring"], sub_category="Streaming Video & Audio", merchant_knowledge_uuid=None)

    def query_filter_by(**kwargs):
        uuid_arg = kwargs.get("transaction_uuid")
        mock_q = MagicMock()
        if uuid_arg == "txn-eur-123":
            mock_q.first.return_value = db_eur
        elif uuid_arg == "txn-fee-123":
            mock_q.first.return_value = db_fee
        elif uuid_arg == "txn-gbp-123":
            mock_q.first.return_value = db_gbp
        else:
            mock_q.first.return_value = None
        return mock_q

    mock_session.query.return_value.filter_by.side_effect = query_filter_by
    mock_session.query.return_value.filter.return_value.first.return_value = None

    with patch.object(service, "_get_or_create_merchant_knowledge") as mock_get_mk:
        mock_mk = MerchantKnowledge(
            knowledge_uuid="mk-recharge-uuid",
            clean_merchant_name="Recharge.com",
            category="Subscriptions & Digital Services",
            sub_category="Streaming Video & Audio",
            tags=["#discretionary", "#online_service"]
        )
        mock_get_mk.return_value = mock_mk

        result = service._match_and_reconcile_cluster(
            session=mock_session,
            user_uuid="user-123",
            bank_uuid="bank-wise",
            cluster=cluster
        )

        assert result is not None
        assert result["merchant_knowledge_uuid"] == "mk-recharge-uuid"
        assert result["foreign_total"] == 12.29
        assert result["foreign_currency"] == "EUR"
        assert result["base_spent"] == 8.64
        assert result["fee_spent"] == 0.03
        assert result["drained_spent"] == 1.94

        # Verify all transactions linked to mk-recharge-uuid
        assert db_eur.merchant_knowledge_uuid == "mk-recharge-uuid"
        assert db_fee.merchant_knowledge_uuid == "mk-recharge-uuid"
        assert db_gbp.merchant_knowledge_uuid == "mk-recharge-uuid"

        # Verify fee tag sanitation
        assert "#fx_fee" in db_fee.tags
        assert "#card_fee" in db_fee.tags
        assert "#penalty" not in db_fee.tags
        assert "#bank_charges" not in db_fee.tags

        # Verify split legs have split payment tags
        assert "#split_payment" in db_eur.tags
        assert "#multi_currency" in db_eur.tags
        assert "#split_payment" in db_gbp.tags
        assert "#multi_currency" in db_gbp.tags


def test_match_and_reconcile_cluster_fee_mismatch_fails():
    service = MultiCurrencyReconciliationService()
    mock_session = MagicMock()

    t0 = datetime(2026, 11, 14, 6, 36, 43, tzinfo=timezone.utc)
    cluster = [
        {
            "transaction_uuid": "txn-fee-1",
            "account_id": "acc-gbp",
            "bank_uuid": "bank-wise",
            "date": t0,
            "amount": -0.50, # Actual fee is 0.50
            "currency": "GBP",
            "description": "Wise Charges for: CARD-999"
        },
        {
            "transaction_uuid": "txn-gbp-1",
            "account_id": "acc-gbp",
            "bank_uuid": "bank-wise",
            "date": t0,
            "amount": -10.00,
            "currency": "GBP",
            "description": "Card transaction of 15.00 EUR issued by Example Merchant (fee: 0.02 GBP)" # Stated fee is 0.02
        }
    ]

    result = service._match_and_reconcile_cluster(
        session=mock_session,
        user_uuid="user-123",
        bank_uuid="bank-wise",
        cluster=cluster
    )
    # Fee mismatch (0.50 vs 0.02) must reject reconciliation
    assert result is None


@pytest.mark.asyncio
async def test_reconcile_transactions_controller():
    from controllers.account_controller import reconcile_multicurrency_transactions

    with patch("services.analytics.multicurrency_reconciliation_service.get_multicurrency_reconciliation_service") as mock_get_svc, \
         patch("utils.cache_utils.clear_user_cache") as mock_clear:
        mock_svc = MagicMock()
        mock_svc.reconcile_user_transactions.return_value = {
            "status": "success",
            "reconciled_groups": 1,
            "reconciled_transactions": 3,
            "message": "Reconciled 3 transactions across 1 multi-currency event."
        }
        mock_get_svc.return_value = mock_svc

        data = await reconcile_multicurrency_transactions("user-test-123")
        assert data["status"] == "success"
        assert data["reconciled_groups"] == 1
        assert data["reconciled_transactions"] == 3
        mock_clear.assert_called()


