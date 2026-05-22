import warnings
import sys
warnings.filterwarnings("ignore")
from mcp.server.fastmcp import FastMCP
from services.mcp_tools.forecaster_tools import generate_financial_forecast, generate_expense_forecast
from services.logger_setup import get_core_logger

logger = get_core_logger("forecaster_server")
mcp = FastMCP("Forecaster")

logger.debug("Registering tool: generate_financial_forecast")
mcp.add_tool(generate_financial_forecast.func)
logger.debug("Registering tool: generate_expense_forecast")
mcp.add_tool(generate_expense_forecast.func)

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--sse", action="store_true", help="Run with SSE transport")
    args = parser.parse_args()

    if args.sse:
        logger.info(f"Starting Forecaster MCP Server with SSE transport")
        mcp.run(transport="sse")
    else:
        logger.info("Starting Forecaster MCP Server")
        try:
            mcp.run()
            logger.info("Forecaster MCP Server running")
        except Exception as e:
            logger.error(f"Failed to run Forecaster MCP Server: {e}")
            raise
