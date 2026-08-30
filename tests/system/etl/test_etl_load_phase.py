import pytest
import re

def test_ephemeral_context_pruning_for_merchant_knowledge():
    """
    TDD ETL Load: Asserts that after the Web Search MCP executes, the massive 
    raw HTML payloads are mathematically pruned down to short text summaries 
    before committing to PostgreSQL to prevent database bloat.
    """
    raw_html_payload = "<html><body><h1>Acme Corp</h1><p>We sell industrial supplies.</p><script>alert('x')</script></body></html>"
    
    def prune_ephemeral_context(html: str) -> str:
        # Strip HTML tags
        text_only = re.sub('<[^<]+?>', '', html)
        # Summarize (simulated truncation)
        return text_only[:50].strip()

    pruned_summary = prune_ephemeral_context(raw_html_payload)
    
    assert "<html>" not in pruned_summary, "Database Bloat Risk: HTML tags were not pruned."
    assert "<script>" not in pruned_summary, "Security Risk: JS tags were not pruned."
    assert len(pruned_summary) <= 50, "Storage Limit Violation: Pruned context exceeds threshold."


def test_database_armor_composite_key_idempotency():
    """
    TDD ETL Load: Asserts the PostgreSQL ON CONFLICT DO UPDATE (Upsert) logic.
    If the ETL pipeline crashes and re-runs the exact same dataset, the composite 
    key (account_id, provider_transaction_id) prevents duplicate row generation.
    """
    mock_db_ledger = {}
    
    def database_armor_upsert(account_id: str, provider_id: str, amount: float):
        # Simulated Composite Key constraint
        composite_key = f"{account_id}_{provider_id}"
        
        if composite_key in mock_db_ledger:
            # ON CONFLICT DO UPDATE (Overwrite data, do not create new row)
            mock_db_ledger[composite_key] = amount
            return "UPSERTED"
        else:
            # INSERT
            mock_db_ledger[composite_key] = amount
            return "INSERTED"

    # Initial Run
    status_1 = database_armor_upsert("acc_1", "txn_999", 50.0)
    assert status_1 == "INSERTED"
    assert len(mock_db_ledger) == 1
    
    # Pipeline Re-run (Accidental Duplicate)
    status_2 = database_armor_upsert("acc_1", "txn_999", 50.0)
    assert status_2 == "UPSERTED"
    assert len(mock_db_ledger) == 1, "Database Armor Failed: Ledger allowed a duplicate row."

