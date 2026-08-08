from services.logger_setup import get_core_logger
import inspect

logger = get_core_logger(__name__)

class MCPBridge:
    """
    A lightweight drop-in replacement that bypasses the old MCP JSON-RPC protocol
    and directly invokes the native Python @tool functions in services/mcp_tools/.
    """
    def __init__(self):
        pass
        
    async def call_tool(self, server_name: str, tool_name: str, arguments: dict):
        try:
            module_name = f"services.mcp_tools.{server_name}_tools"
            module = __import__(module_name, fromlist=[tool_name])
            
            tool_func = getattr(module, tool_name)
            
            if hasattr(tool_func, "invoke"):
                return tool_func.invoke(arguments)
                
            if inspect.iscoroutinefunction(tool_func):
                return await tool_func(**arguments)
            else:
                return tool_func(**arguments)
                
        except Exception as e:
            logger.error(f"Native Tool Execution Error [{server_name}.{tool_name}]: {e}")
            return f"Error executing tool: {str(e)}"
