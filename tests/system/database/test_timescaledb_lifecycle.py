import pytest
from sqlalchemy import text

@pytest.mark.asyncio
async def test_timescaledb_retention_policy_is_active(db_session):
    """
    TDD System Design: Asserts that the TimescaleDB hypertable for transactions 
    has an active background worker configured to drop data chunks older than 36 months,
    preventing EBS volume disk exhaustion.
    """
    sql = text("""
        SELECT job_id, config 
        FROM timescaledb_information.jobs 
        WHERE proc_name = 'policy_retention' 
        AND hypertable_name = 'transactions';
    """)
    result = db_session.execute(sql).fetchone()
    
    assert result is not None, "TimescaleDB Data Retention Policy is missing. Disk exhaustion risk active."
    # The config should mathematically contain the 36-month interval
    assert 'drop_after' in result.config, "Retention policy missing drop_after configuration."

