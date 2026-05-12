import logging
from mcp.server.fastmcp import FastMCP
import random

logger = logging.getLogger("uvicorn.error")
mcp = FastMCP("Macro")


@mcp.tool()
def get_geopolitical_sentiment() -> str:
    val = random.uniform(-0.5, 0.5)
    sentiment = "positive" if val > 0 else "negative"
    return f"Global sentiment is currently {sentiment}."


@mcp.tool()
def get_commodity_spot_prices() -> str:
    return "Oil: 80.50 USD/bbl, Gold: 2000.00 USD/oz"


if __name__ == "__main__":
    mcp.run()
