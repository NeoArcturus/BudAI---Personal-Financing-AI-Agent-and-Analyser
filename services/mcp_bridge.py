import os
import json
import asyncio
from datetime import datetime
from services.logger_setup import get_core_logger
import iii

logger = get_core_logger("mcp_bridge")

III_URL = os.getenv("III_URL", "ws://iii-engine:49134")

class MCPBridge:
    def __init__(self):
        self.workspace_dir = os.path.join(
            os.path.expanduser("~"), "BudAI_Workspace")
        self._ensure_directories()
        self.iii_client = iii.register_worker(III_URL)

    def _ensure_directories(self):
        dirs = ["ingestion", "rules", "backups", "advisory_state", "exports"]
        for d in dirs:
            dir_path = os.path.join(self.workspace_dir, d)
            os.makedirs(dir_path, exist_ok=True)
            
        rules_path = os.path.join(
            self.workspace_dir, "rules", "budai_rules.json")
        if not os.path.exists(rules_path):
            with open(rules_path, "w") as f:
                json.dump({"custom_rules": {}}, f)

    async def call_iii_tool(self, worker_name: str, function_name: str, tool_args: dict):
        logger.info(f"--- III REQUEST: [{worker_name}::{function_name}] ---")
        try:
            result = await asyncio.to_thread(
                self.iii_client.trigger,
                {
                    "function_id": f"{worker_name}::{function_name}",
                    "payload": tool_args
                }
            )
            result_text = str(result)
            log_text = result_text if len(result_text) < 1000 else result_text[:1000] + "... [TRUNCATED]"
            logger.info(f"--- III RESPONSE: [{worker_name}::{function_name}] ---")
            logger.info(f"RESULT: {log_text}")
            return result_text
        except Exception as e:
            logger.error(f"--- III ERROR: [{worker_name}::{function_name}] ---")
            logger.error(f"EXCEPTION: {str(e)}", exc_info=True)
            raise

    def call_iii_tool_sync(self, worker_name: str, function_name: str, tool_args: dict):
        return self.iii_client.trigger(
            {
                "function_id": f"{worker_name}::{function_name}",
                "payload": tool_args
            }
        )

    def write_advisory_file(self, user_uuid: str, chart_type: str, raw_data: dict, ai_analysis: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"Advisory_{user_uuid}_{chart_type}_{timestamp}.json"
        file_path = os.path.join(
            self.workspace_dir, "advisory_state", file_name)
        payload = {
            "metadata": {"user_uuid": user_uuid, "generated_at": timestamp, "chart_type": chart_type},
            "raw_data": raw_data,
            "ai_analysis": ai_analysis
        }
        try:
            with open(file_path, "w") as f:
                json.dump(payload, f, indent=2)
            return file_path
        except Exception as e:
            logger.error(f"Failed to write advisory file: {e}")
            raise

    def read_user_rules(self):
        file_path = os.path.join(self.workspace_dir, "rules", "budai_rules.json")
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception:
            return {"custom_rules": {}}

    def generate_outbound_statement(self, file_path: str, data_payload: str) -> str:
        try:
            with open(file_path, "w") as f:
                f.write(data_payload)
            return "File exported successfully."
        except Exception as e:
            logger.error(f"Export failed for {file_path}: {str(e)}")
            return f"Export failed: {str(e)}"
