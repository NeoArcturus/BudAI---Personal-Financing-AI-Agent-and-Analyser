import os
import json
import asyncio
from datetime import datetime
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
from services.logger_setup import get_core_logger

logger = get_core_logger("mcp_bridge")


class MCPBridge:
    def __init__(self):
        self.workspace_dir = os.path.join(
            os.path.expanduser("~"), "BudAI_Workspace")
        self._ensure_directories()

    def _ensure_directories(self):
        logger.debug("Ensuring directories exist")
        dirs = ["ingestion", "rules", "backups", "advisory_state", "exports"]
        for d in dirs:
            dir_path = os.path.join(self.workspace_dir, d)
            os.makedirs(dir_path, exist_ok=True)
            
        rules_path = os.path.join(
            self.workspace_dir, "rules", "budai_rules.json")
        if not os.path.exists(rules_path):
            with open(rules_path, "w") as f:
                json.dump({"custom_rules": {}}, f)

    async def call_sse_tool(self, server_url: str, tool_name: str, tool_args: dict):
        logger.info(f"Calling SSE tool {tool_name} at {server_url}")
        async with sse_client(server_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=tool_args)
                return result.content[0].text

    def _execute_mcp_tool(self, command: str, args: list, tool_name: str, tool_args: dict, env: dict = None):
        # ... (keeping for backward compatibility or direct stdio fallback)
        logger.debug(f"Executing MCP tool: {tool_name} with command: {command}")
        
        async def _run():
            server_params = StdioServerParameters(
                command=command, args=args, env=env)
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    logger.debug("Initializing MCP session")
                    await session.initialize()
                    logger.debug(f"Calling MCP tool: {tool_name}")
                    result = await session.call_tool(tool_name, arguments=tool_args)
                    logger.debug(f"MCP tool {tool_name} execution complete")
                    return result
                    
        try:
            loop = asyncio.get_running_loop()
            is_running = loop.is_running()
            logger.debug(f"Event loop is running: {is_running}")
        except RuntimeError:
            is_running = False
            logger.debug("No event loop running")

        if is_running:
            logger.debug("Running MCP tool in ThreadPoolExecutor")
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                try:
                    return pool.submit(asyncio.run, _run()).result()
                except Exception as e:
                    logger.error(f"MCP Tool Execution Failed in ThreadPool: {e}")
                    raise
        else:
            logger.debug("Running MCP tool with asyncio.run")
            try:
                return asyncio.run(_run())
            except Exception as e:
                logger.error(f"MCP Tool Execution Failed in asyncio.run: {e}")
                raise

    def write_advisory_file(self, user_uuid: str, chart_type: str, raw_data: dict, ai_analysis: str) -> str:
        logger.info(f"Writing advisory file for user {user_uuid}, type {chart_type}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"Advisory_{user_uuid}_{chart_type}_{timestamp}.json"
        file_path = os.path.join(
            self.workspace_dir, "advisory_state", file_name)
        
        payload = {
            "metadata": {
                "user_uuid": user_uuid,
                "generated_at": timestamp,
                "chart_type": chart_type
            },
            "raw_data": raw_data,
            "ai_analysis": ai_analysis
        }
        
        logger.debug(f"Advisory payload generated: {file_name}")
        
        try:
            self._execute_mcp_tool(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem",
                      self.workspace_dir],
                tool_name="write_file",
                tool_args={"path": file_path,
                           "content": json.dumps(payload, indent=2)}
            )
            logger.info(f"Advisory file successfully written to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to write advisory file: {e}")
            raise

    def read_user_rules(self):
        logger.info("Reading user rules")
        file_path = os.path.join(
            self.workspace_dir, "rules", "budai_rules.json")
        try:
            result = self._execute_mcp_tool(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem",
                      self.workspace_dir],
                tool_name="read_file",
                tool_args={"path": file_path}
            )
            content = result.content[0].text
            rules = json.loads(content)
            logger.debug("Successfully parsed user rules")
            return rules
        except Exception as e:
            logger.error(f"Failed to read user rules: {e}")
            return {"custom_rules": {}}

    def generate_outbound_statement(self, file_path: str, data_payload: str) -> str:
        logger.info(f"Generating outbound statement at {file_path}")
        try:
            result = self._execute_mcp_tool(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem",
                      self.workspace_dir],
                tool_name="write_file",
                tool_args={"path": file_path, "content": data_payload}
            )
            logger.info(f"Generated outbound statement successful at {file_path}")
            return result.content[0].text
        except Exception as e:
            logger.error(f"Export failed for {file_path}: {str(e)}")
            return f"Export failed: {str(e)}"

    def interact_with_memory(self, tool_name: str, tool_args: dict) -> str:
        logger.info(f"Interacting with memory tool: {tool_name}")
        try:
            result = self._execute_mcp_tool(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-memory"],
                tool_name=tool_name,
                tool_args=tool_args
            )
            logger.info(f"Memory interaction successful for {tool_name}")
            return result.content[0].text
        except Exception as e:
            logger.error(f"Memory operation {tool_name} failed: {str(e)}")
            return f"Memory operation failed: {str(e)}"
