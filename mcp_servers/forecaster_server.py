import logging
from mcp.server.fastmcp import FastMCP
from services.mcp_tools.internal_tools import generate_financial_forecast, generate_expense_forecast

logger = logging.getLogger("uvicorn.error")
mcp = FastMCP("Forecaster")

mcp.add_tool(generate_financial_forecast.func)
mcp.add_tool(generate_expense_forecast.func)

if __name__ == "__main__":
    logger.info("Initializing Forecaster MCP Server")
    mcp.run()
