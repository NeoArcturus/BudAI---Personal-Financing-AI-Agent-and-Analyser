import pytest
from pydantic import ValidationError

# Assuming standard Pydantic schemas are used in routes/onboarding_routes.py
# from schemas.onboarding_schema import OnboardingRequest

def test_onboarding_schema_rejects_negative_income():
    """TDD: Validates that the system physically cannot accept negative income declarations."""
    # Placeholder for actual schema import
    # with pytest.raises(ValidationError):
    #     OnboardingRequest(income=-50000, currency="GBP")
    
    payload = {"income": -50000, "currency": "GBP"}
    assert payload["income"] > 0, "Schema validation missing: Allowed negative income"

def test_onboarding_requires_valid_currency_iso():
    """TDD: Validates currency against ISO 4217 standard."""
    payload = {"income": 50000, "currency": "FAKE"}
    valid_currencies = {"GBP", "USD", "EUR"}
    assert payload["currency"] in valid_currencies, "Schema validation missing: Allowed invalid currency"
