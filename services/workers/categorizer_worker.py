import re
import asyncio
import logging
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from services.logger_setup import get_core_logger
from services.mcp_bridge import MCPBridge

logger = get_core_logger(__name__)

class WorkerReasoningCallback(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs) -> None:
        try:
            content = response.generations[0][0].message.content
            if content:
                match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
                if match: logger.info(f"Worker Reasoning: {match.group(1).strip()}")
                elif content.strip(): logger.info(f"Worker Thoughts: {content.strip()}")
        except Exception: pass

async def run_categorizer_worker(state):
    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
    if not base_url.endswith("/v1"): base_url = f"{base_url}/v1"
    llm = ChatOpenAI(model="mlx-community/Qwen2.5-7B-Instruct-4bit", base_url=base_url, api_key="budai-local", temperature=0, callbacks=[WorkerReasoningCallback()])
    bridge = MCPBridge()

    @tool
    async def classify_financial_data_wrapper(from_date: str, to_date: str, account_ids: list[str]) -> str:
        """Categorize transactions for selected accounts."""
        return await bridge.call_iii_tool("categorizer", "classify_financial_data", {"from_date": from_date, "to_date": to_date, "account_ids": account_ids, "user_uuid": state['user_uuid']})

    @tool
    async def create_bargraph_chart_and_save_wrapper(account_ids: list[str]) -> str:
        """Create bar chart of categories for selected accounts."""
        return await bridge.call_iii_tool("categorizer", "create_bargraph_chart_and_save", {"account_ids": account_ids, "user_uuid": state['user_uuid']})

    tools = [classify_financial_data_wrapper, create_bargraph_chart_and_save_wrapper]
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a robotic data routing node for categorization.
         RULES:
         1. You MUST use a tool to answer the user's query.
         2. Pass a list of account IDs or names.
         3. NO 'ALL': Never pass "ALL" as an account identifier. Use only the specific IDs or names provided in the human message.
         4. MULTI-ACCOUNT: If multiple accounts are provided, categorize data across all of them.
         """),
        ("human", "User Query: {input}\nUser ID: {user_uuid}\nSelected Accounts: {active_account_id}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True, return_intermediate_steps=True)
    selected_ids = state['active_account_id'].split(",") if state['active_account_id'] else []
    result = await agent_executor.ainvoke({"input": state['user_input'], "user_uuid": state['user_uuid'], "active_account_id": selected_ids})
    output = str(result["intermediate_steps"][-1][1]) if "intermediate_steps" in result and result["intermediate_steps"] else result.get("output", "")
    cache_id, chart_type = None, None
    if output:
        match = re.search(r'\[TRIGGER_([A-Z_]+):([^\]]+)\]', output)
        if match:
            chart_type = match.group(1)
            cache_id = match.group(2).split(':')[0]
            output = re.sub(r'\[TRIGGER_[A-Z_]+:[^\]]+\]', '', output).strip()
    return {"worker_summary": output, "cache_id": cache_id, "chart_type": chart_type}
