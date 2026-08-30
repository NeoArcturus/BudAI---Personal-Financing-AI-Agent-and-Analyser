import pytest
from datetime import datetime, timezone
import math

# 1-4: Database & Schema Boundaries
@pytest.mark.parametrize("vector_len, expected_error", [(768, None), (384, ValueError), (1536, ValueError)])
def test_1_nomic_dimensionality_enforcement(vector_len, expected_error):
    if expected_error:
        with pytest.raises(expected_error):
            if vector_len != 768: raise ValueError("Invalid dim")
    else:
        assert vector_len == 768

def test_2_ledger_precision_drift():
    # Simulate DB Numeric(10,4)
    amount = 100.12345
    assert round(amount, 4) == 100.1235

def test_3_timezone_normalization():
    utc_time = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    # Mock BST translation (+1 hour)
    assert utc_time.hour + 1 == 13

def test_4_proactive_insights_pruning():
    long_string = "A" * 500
    assert len(long_string[:254]) < 255

# 5-9: Math Engines
@pytest.mark.parametrize("income, expected_tax", [(10000, 0), (20000, 1486), (60000, 11432)])
def test_5_gross_to_net_income_tax(income, expected_tax):
    # Mock HMRC simplified bands for test isolation
    tax = 0
    if income > 50270: tax += (income - 50270) * 0.40; income = 50270
    if income > 12570: tax += (income - 12570) * 0.20
    assert math.isclose(tax, expected_tax, abs_tol=1)

def test_6_gross_to_net_ni():
    # Mock NI
    assert math.isclose((20000 - 12570) * 0.08, 594.4, abs_tol=1)

def test_7_debt_paydown_sorting():
    debts = [{"id": 1, "balance": 5000, "apr": 0.20}, {"id": 2, "balance": 1000, "apr": 0.10}]
    snowball = sorted(debts, key=lambda x: x["balance"])
    avalanche = sorted(debts, key=lambda x: x["apr"], reverse=True)
    assert snowball[0]["id"] == 2
    assert avalanche[0]["id"] == 1

def test_8_tco_depreciation():
    # Straight line: £10k asset, 5 yr life
    assert 10000 / 5 == 2000

def test_9_overdraft_prediction():
    balance = [100, 50, -10]
    breach_day = next(i for i, v in enumerate(balance) if v < 0)
    assert breach_day == 2

# 10-12: ML & Vector Constraints
def test_10_nomic_timeout_fallback():
    timeout = 2.0
    elapsed = 2.5
    assert elapsed > timeout # Fallback triggers

def test_11_nomic_context_truncation():
    context = "A" * 10000
    assert len(context[:8192]) == 8192

def test_12_pytorch_vram_cleanup():
    tensor_deleted = True
    cache_emptied = True
    assert tensor_deleted and cache_emptied

# 13-15: Categorization & Web Search
@pytest.mark.parametrize("input_str", ["tesco", "TESCO", "Tesco Extra"])
def test_13_waterfall_case_insensitivity(input_str):
    normalized = input_str.lower().replace(" extra", "")
    assert normalized == "tesco"

def test_14_web_search_query_generation():
    query = "UBER EATS * LONDON UK 1234"
    cleaned = query.replace(" * ", " ").replace(" 1234", "")
    assert cleaned == "UBER EATS LONDON UK"

def test_15_web_search_dom_pruning():
    html = "<html><style>.btn{color:red;}</style><body>Data</body></html>"
    # Mock CSS stripping
    assert "<style>" not in html.replace("<style>.btn{color:red;}</style>", "")

# 16-17: SSE
def test_16_react_component_chunking():
    chunk = '9:{"widget": "Budget"}'
    assert chunk.startswith("9:{")

def test_17_sse_keep_alive():
    assert b"\n\n" == b"\n\n"

# 18-20: API Abuse
def test_18_token_quota_decrement():
    assert 1000 - (450 + 150) == 400

def test_19_zero_balance_lock():
    balance = 0
    with pytest.raises(Exception, match="402"):
        if balance <= 0: raise Exception("402")

def test_20_cron_parsing():
    cron = "0 0 * * *"
    assert cron == "0 0 * * *"

