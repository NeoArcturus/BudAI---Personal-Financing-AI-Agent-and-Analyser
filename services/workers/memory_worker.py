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
from services.mcp_tools.external_tools import search_user_memory, save_to_user_memory

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

async def run_memory_worker(state):
    logger.info(f"Received task: {state['user_input']}")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
    if not base_url.endswith("/v1"): base_url = f"{base_url}/v1"
    llm = ChatOpenAI(model="mlx-community/Qwen2.5-7B-Instruct-4bit", base_url=base_url, api_key="budai-local", temperature=0, callbacks=[WorkerReasoningCallback()])
    bridge = MCPBridge()

    @tool
    async def search_financial_history_semantic_wrapper(query: str) -> str:
        """Perform a semantic search on the user's transaction history."""
        return await bridge.call_iii_tool("memory", "search_financial_history_semantic", {"query": query, "user_uuid": state['user_uuid']})

    @tool
    async def get_seasonal_behavior_context_wrapper() -> str:
        """Retrieves a summary of how the user historically behaves in the current month."""
        return await bridge.call_iii_tool("memory", "get_seasonal_behavior_context", {"user_uuid": state['user_uuid']})

    tools = [search_financial_history_semantic_wrapper, get_seasonal_behavior_context_wrapper, search_user_memory, save_to_user_memory]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the Knowledge & History Agent. Think step-by-step."),
        ("human", "User Query: {input}\nUser ID: {user_uuid}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True, return_intermediate_steps=True)
    result = await agent_executor.ainvoke({"input": state['user_input'], "user_uuid": state['user_uuid']})
    output = str(result["intermediate_steps"][-1][1]) if "intermediate_steps" in result and result["intermediate_steps"] else result.get("output", "")
    return {"worker_summary": output, "cache_id": None, "chart_type": None}
