import logging
from mcp.server.fastmcp import FastMCP
from services.mcp_tools.internal_tools import plot_expenses, find_total_spent_for_given_category, find_highest_spending_category, plot_cash_flow_mixed
from services.mcp_tools.external_tools import export_custom_statement

logger = logging.getLogger("uvicorn.error")
mcp = FastMCP("Analyser")

mcp.add_tool(plot_expenses.func)
mcp.add_tool(find_total_spent_for_given_category.func)
mcp.add_tool(find_highest_spending_category.func)
mcp.add_tool(plot_cash_flow_mixed.func)
mcp.add_tool(export_custom_statement.func)

if __name__ == "__main__":
    logger.info("Initializing Analyser MCP Server")
    mcp.run()
