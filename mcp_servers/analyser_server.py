import warnings
import sys
warnings.filterwarnings("ignore")
import logging
from mcp.server.fastmcp import FastMCP
from services.mcp_tools.analyser_tools import plot_expenses, find_total_spent_for_given_category, find_highest_spending_category, plot_cash_flow_mixed
from services.mcp_tools.external_tools import export_custom_statement
from services.mcp_tools.shared_utils import _cache_chart_data # For any direct use
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)
mcp = FastMCP("Analyser")

logger.info("Registering tools for Analyser MCP Server")
mcp.add_tool(plot_expenses.func)
mcp.add_tool(find_total_spent_for_given_category.func)
mcp.add_tool(find_highest_spending_category.func)
mcp.add_tool(plot_cash_flow_mixed.func)
mcp.add_tool(export_custom_statement.func)

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--sse", action="store_true", help="Run with SSE transport")
    args = parser.parse_args()

    if args.sse:
        logger.info(f"Starting Analyser MCP Server with SSE transport")
        mcp.run(transport="sse")
    else:
        logger.info("Initializing Analyser MCP Server")
        mcp.run()
        logger.info("Analyser MCP Server has stopped")
