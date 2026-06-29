from pydantic import BaseModel, Field


class MarketComparisonInput(BaseModel):
    user_uuid: str = Field(..., description="The user UUID.")
    category: str = Field(
        ..., description="The expense category to analyze (e.g., 'Bills & Utilities', 'Entertainment').")
    ticker: str = Field(
        ..., description="The Yahoo Finance ticker to compare against (e.g., 'NG=F', '^GSPC').")
    days: int = Field(default=180, description="The lookback period in days.")


class HistoricalMarketDataInput(BaseModel):
    ticker: str = Field(..., description="The Yahoo Finance ticker.")
    period: str = Field(
        default="6mo", description="The period (1mo, 6mo, 1y).")
    interval: str = Field(default="1d", description="The interval (1d, 1wk).")


class CurrencyConversionInput(BaseModel):
    amount: float = Field(..., description="The amount to convert.")
    from_currency: str = Field(..., description="The source currency code (e.g., 'USD', 'EUR').")
    to_currency: str = Field(..., description="The target currency code (e.g., 'GBP', 'INR').")
