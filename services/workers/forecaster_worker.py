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

async def run_forecaster_worker(state):
    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
    if not base_url.endswith("/v1"): base_url = f"{base_url}/v1"
    llm = ChatOpenAI(model="mlx-community/Qwen2.5-7B-Instruct-4bit", base_url=base_url, api_key="budai-local", temperature=0, callbacks=[WorkerReasoningCallback()])
    bridge = MCPBridge()

    @tool
    async def generate_expense_forecast_wrapper(account_ids: list[str], days: int) -> str:
        """Forecast future expenses for selected accounts."""
        return await bridge.call_iii_tool("forecaster", "generate_expense_forecast", {"account_ids": account_ids, "days": days, "user_uuid": state['user_uuid']})

    @tool
    async def generate_financial_forecast_wrapper(account_ids: list[str], days: int, discipline_multiplier: float) -> str:
        """Forecast overall financial balance for selected accounts."""
        return await bridge.call_iii_tool("forecaster", "generate_financial_forecast", {"account_ids": account_ids, "days": days, "discipline_multiplier": discipline_multiplier, "user_uuid": state['user_uuid']})

    tools = [generate_expense_forecast_wrapper, generate_financial_forecast_wrapper]
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a robotic data routing node for forecasting.
         RULES:
         1. You MUST use a tool to answer the user's query.
         2. EXTRACTION: Prioritize extracting specific bank names from the "User Query". If the user mentions "Wise", pass ["Wise"].
         3. CONTEXT FALLBACK: Only use the "Selected Accounts" if the "User Query" is vague and does not mention any specific bank.
         4. NO 'ALL': Never pass "ALL" as an account identifier to a tool. If both query and context are vague, use the list of available IDs from the context, but never the string "ALL".
         5. MULTI-ACCOUNT: If multiple accounts are mentioned, pass them as a list.
         6. GBP ONLY: Always think and reason in GBP (£). Never use Dollars ($).
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
