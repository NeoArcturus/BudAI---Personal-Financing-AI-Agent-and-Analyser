import logging
from mcp.server.fastmcp import FastMCP
from services.mcp_tools.internal_tools import analyze_critical_survival_metrics, analyze_wealth_acceleration_metrics, plot_health_radar

logger = logging.getLogger("uvicorn.error")
mcp = FastMCP("Health")

mcp.add_tool(analyze_critical_survival_metrics.func)
mcp.add_tool(analyze_wealth_acceleration_metrics.func)
mcp.add_tool(plot_health_radar.func)

if __name__ == "__main__":
    logger.info("Initializing Health MCP Server")
    mcp.run()
