import os
import json
import asyncio
import logging
from datetime import datetime
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

logger = logging.getLogger("uvicorn.error")


class MCPBridge:
    def __init__(self):
        self.workspace_dir = os.path.join(
            os.path.expanduser("~"), "BudAI_Workspace")
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), ".."))
        self.mcp_servers_dir = os.path.join(self.project_root, "mcp_servers")
        self._ensure_directories()

    def _ensure_directories(self):
        dirs = ["ingestion", "rules", "backups", "advisory_state", "exports"]
        for d in dirs:
            os.makedirs(os.path.join(self.workspace_dir, d), exist_ok=True)
        rules_path = os.path.join(
            self.workspace_dir, "rules", "budai_rules.json")
        if not os.path.exists(rules_path):
            with open(rules_path, "w") as f:
                json.dump({"custom_rules": {}}, f)
        os.makedirs(self.mcp_servers_dir, exist_ok=True)

    def _execute_mcp_tool(self, command: str, args: list, tool_name: str, tool_args: dict, env: dict = None):
        async def _run():
            server_params = StdioServerParameters(
                command=command, args=args, env=env)
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=tool_args)
                    return result
        try:
            loop = asyncio.get_running_loop()
            is_running = loop.is_running()
        except RuntimeError:
            is_running = False

        if is_running:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                try:
                    return pool.submit(asyncio.run, _run()).result()
                except Exception as e:
                    logger.error(f"MCP Tool Execution Failed: {e}")
                    raise
        else:
            try:
                return asyncio.run(_run())
            except Exception as e:
                logger.error(f"MCP Tool Execution Failed: {e}")
                raise

    def write_advisory_file(self, user_uuid: str, chart_type: str, raw_data: dict, ai_analysis: str) -> str:
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
        self._execute_mcp_tool(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem",
                  self.workspace_dir],
            tool_name="write_file",
            tool_args={"path": file_path,
                       "content": json.dumps(payload, indent=2)}
        )
        logger.info(f"Advisory file written to {file_path}")
        return file_path

    def read_user_rules(self):
        file_path = os.path.join(
            self.workspace_dir, "rules", "budai_rules.json")
        result = self._execute_mcp_tool(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem",
                  self.workspace_dir],
            tool_name="read_file",
            tool_args={"path": file_path}
        )
        try:
            content = result.content[0].text
            return json.loads(content)
        except Exception as e:
            logger.error(f"Failed to read user rules: {e}")
            return {"custom_rules": {}}

    def generate_outbound_statement(self, file_path: str, data_payload: str) -> str:
        result = self._execute_mcp_tool(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem",
                  self.workspace_dir],
            tool_name="write_file",
            tool_args={"path": file_path, "content": data_payload}
        )
        try:
            logger.info(f"Generated outbound statement at {file_path}")
            return result.content[0].text
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            return f"Export failed: {str(e)}"

    def interact_with_memory(self, tool_name: str, tool_args: dict) -> str:
        result = self._execute_mcp_tool(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-memory"],
            tool_name=tool_name,
            tool_args=tool_args
        )
        try:
            logger.info(f"Memory interaction successful for {tool_name}")
            return result.content[0].text
        except Exception as e:
            logger.error(f"Memory operation failed: {str(e)}")
            return f"Memory operation failed: {str(e)}"
