import warnings
import sys
warnings.filterwarnings("ignore")
import logging
from mcp.server.fastmcp import FastMCP
from services.mcp_tools.categorizer_tools import classify_financial_data, create_bargraph_chart_and_save, create_pie_chart_and_save, update_transaction_category, retrain_categorization_model
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)
mcp = FastMCP("Categorizer")

logger.info("Registering tools for Categorizer MCP Server")
mcp.add_tool(classify_financial_data.func)
mcp.add_tool(create_bargraph_chart_and_save.func)
mcp.add_tool(create_pie_chart_and_save.func)
mcp.add_tool(update_transaction_category.func)
mcp.add_tool(retrain_categorization_model.func)

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--sse", action="store_true", help="Run with SSE transport")
    args = parser.parse_args()

    if args.sse:
        logger.info(f"Starting Categorizer MCP Server with SSE transport")
        mcp.run(transport="sse")
    else:
        logger.info("Initializing Categorizer MCP Server")
        mcp.run()
        logger.info("Categorizer MCP Server has stopped")
