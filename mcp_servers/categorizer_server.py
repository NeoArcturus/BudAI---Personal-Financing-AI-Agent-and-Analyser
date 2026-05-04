import logging
from mcp.server.fastmcp import FastMCP
from services.mcp_tools.internal_tools import classify_financial_data, create_bargraph_chart_and_save, create_pie_chart_and_save, update_transaction_category, retrain_categorization_model

logger = logging.getLogger("uvicorn.error")
mcp = FastMCP("Categorizer")

mcp.add_tool(classify_financial_data.func)
mcp.add_tool(create_bargraph_chart_and_save.func)
mcp.add_tool(create_pie_chart_and_save.func)
mcp.add_tool(update_transaction_category.func)
mcp.add_tool(retrain_categorization_model.func)

if __name__ == "__main__":
    logger.info("Initializing Categorizer MCP Server")
    mcp.run()
