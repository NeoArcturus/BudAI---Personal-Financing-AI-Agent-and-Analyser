import json
from typing import Dict, Any, Optional
import yfinance as yf
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)


def fetch_spot_price(ticker_symbol: str) -> Optional[float]:
    """
    Fetches the latest spot rate or market close for a given Yahoo Finance ticker.
    Uses fast_info attributes with historical candle fallback.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        price = ticker.fast_info.get("lastPrice") or getattr(ticker.fast_info, "last_price", None)
        if price and float(price) > 0:
            return float(price)
        
        hist = ticker.history(period="1d")
        if not hist.empty:
            val = float(hist["Close"].iloc[-1])
            if val > 0:
                return val
    except Exception as e:
        logger.debug(json.dumps({"message": f"Spot rate lookup failed for {ticker_symbol}: {e}", "status_code": 400}))
    return None


def get_usd_base_rate(currency: str) -> Optional[float]:
    """
    Resolves the base exchange rate of 1 USD in target currency units (USD -> Currency).
    Checks USD{C}=X and inverse {C}USD=X.
    """
    c = currency.upper().strip()
    if c == "USD":
        return 1.0

    # 1. Direct pair: 1 USD = X Currency (e.g. USDJPY=X)
    direct_usd = fetch_spot_price(f"USD{c}=X")
    if direct_usd:
        return direct_usd

    # 2. Inverted pair: 1 Currency = Y USD (e.g. GBPUSD=X -> 1 USD = 1/Y GBP)
    inv_usd = fetch_spot_price(f"{c}USD=X")
    if inv_usd and inv_usd > 0:
        return 1.0 / inv_usd

    return None


def get_cross_fx_rate(from_currency: str, to_currency: str, amount: float = 1.0) -> Dict[str, Any]:
    """
    Calculates live FX conversion rate between two currencies.
    Attempts direct market pair first; if unavailable, triangulates against USD base reference.
    """
    from_curr = from_currency.strip().upper()
    to_curr = to_currency.strip().upper()

    if not from_curr.isalpha() or not to_curr.isalpha():
        raise ValueError(f"Invalid currency codes provided: from='{from_currency}', to='{to_currency}'")

    if from_curr == to_curr:
        return {
            "from_currency": from_curr,
            "to_currency": to_curr,
            "rate": 1.0,
            "amount": round(amount, 4),
            "converted_amount": round(amount, 4),
            "triangulated": False,
            "reference_currency": "DIRECT"
        }

    # Attempt direct FX pair
    direct_symbol = f"{from_curr}{to_curr}=X"
    direct_rate = fetch_spot_price(direct_symbol)
    if direct_rate:
        converted = amount * direct_rate
        return {
            "from_currency": from_curr,
            "to_currency": to_curr,
            "rate": round(direct_rate, 6),
            "amount": round(amount, 4),
            "converted_amount": round(converted, 4),
            "triangulated": False,
            "reference_currency": "DIRECT"
        }

    # Fallback to USD Triangulation
    rate_usd_from = get_usd_base_rate(from_curr)
    if not rate_usd_from:
        raise ValueError(f"Unable to resolve baseline USD rate for origin currency '{from_curr}'")

    rate_usd_to = get_usd_base_rate(to_curr)
    if not rate_usd_to:
        raise ValueError(f"Unable to resolve baseline USD rate for destination currency '{to_curr}'")

    # Cross rate: (USD -> TO) / (USD -> FROM) = FROM -> TO
    cross_rate = rate_usd_to / rate_usd_from
    converted = amount * cross_rate

    return {
        "from_currency": from_curr,
        "to_currency": to_curr,
        "rate": round(cross_rate, 6),
        "amount": round(amount, 4),
        "converted_amount": round(converted, 4),
        "triangulated": True,
        "reference_currency": "USD",
        "triangulation_details": {
            f"USD_{from_curr}": round(rate_usd_from, 6),
            f"USD_{to_curr}": round(rate_usd_to, 6)
        }
    }
