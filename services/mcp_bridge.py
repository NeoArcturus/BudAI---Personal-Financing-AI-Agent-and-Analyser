import json
import importlib
import pkgutil
from services.logger_setup import get_core_logger
import inspect
import services.mcp_tools

logger = get_core_logger(__name__)

class MCPBridge:
    """
    A lightweight drop-in replacement that bypasses the old MCP JSON-RPC protocol
    and directly invokes the native Python @tool functions in services/mcp_tools/.
    """
    def __init__(self):
        self._tool_cache = {}
        
    def _find_tool_module(self, tool_name: str, preferred_server: str = None) -> str:
        if tool_name in self._tool_cache:
            return self._tool_cache[tool_name]
            
        if preferred_server:
            try:
                mod_name = f"services.mcp_tools.{preferred_server}_tools"
                mod = importlib.import_module(mod_name)
                if hasattr(mod, tool_name):
                    self._tool_cache[tool_name] = mod_name
                    return mod_name
            except ImportError:
                pass
                
        # Scan all modules in mcp_tools if not found in the preferred location
        for _, module_name, _ in pkgutil.iter_modules(services.mcp_tools.__path__):
            full_module_name = f"services.mcp_tools.{module_name}"
            try:
                mod = importlib.import_module(full_module_name)
                if hasattr(mod, tool_name):
                    self._tool_cache[tool_name] = full_module_name
                    return full_module_name
            except Exception:
                continue
                
        return None

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict):
        try:
            module_name = self._find_tool_module(tool_name, server_name)
            
            if not module_name:
                raise AttributeError(f"Could not find tool '{tool_name}' in any module inside services/mcp_tools")
                
            module = importlib.import_module(module_name)
            tool_func = getattr(module, tool_name)
            
            if hasattr(tool_func, "invoke"):
                return tool_func.invoke(arguments)
                
            if inspect.iscoroutinefunction(tool_func):
                return await tool_func(**arguments)
            else:
                return tool_func(**arguments)
                
        except Exception as e:
            logger.error(json.dumps({"message": f"Native Tool Execution Error [{server_name}.{tool_name}]: {e}", "status_code": 500}))
            return f"Error executing tool: {str(e)}"
