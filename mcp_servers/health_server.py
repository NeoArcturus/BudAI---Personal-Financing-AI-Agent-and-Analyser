import warnings
import sys
warnings.filterwarnings("ignore")
from mcp.server.fastmcp import FastMCP
from services.mcp_tools.health_tools import analyze_critical_survival_metrics, analyze_wealth_acceleration_metrics, plot_health_radar
from services.logger_setup import get_core_logger

logger = get_core_logger("health_server")
mcp = FastMCP("Health")

logger.debug("Registering tool: analyze_critical_survival_metrics")
mcp.add_tool(analyze_critical_survival_metrics.func)
logger.debug("Registering tool: analyze_wealth_acceleration_metrics")
mcp.add_tool(analyze_wealth_acceleration_metrics.func)
logger.debug("Registering tool: plot_health_radar")
mcp.add_tool(plot_health_radar.func)

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--sse", action="store_true", help="Run with SSE transport")
    args = parser.parse_args()

    if args.sse:
        logger.info(f"Starting Health MCP Server with SSE transport")
        mcp.run(transport="sse")
    else:
        logger.info("Starting Health MCP Server")
        try:
            mcp.run()
            logger.info("Health MCP Server running")
        except Exception as e:
            logger.error(f"Failed to run Health MCP Server: {e}")
            raise
