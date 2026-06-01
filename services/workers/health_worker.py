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

async def run_health_worker(state):
    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
    if not base_url.endswith("/v1"): base_url = f"{base_url}/v1"
    llm = ChatOpenAI(model="mlx-community/Qwen2.5-7B-Instruct-4bit", base_url=base_url, api_key="budai-local", temperature=0, callbacks=[WorkerReasoningCallback()])
    bridge = MCPBridge()

    @tool
    async def analyze_wealth_acceleration_metrics_wrapper() -> str:
        """Analyze wealth and growth."""
        return await bridge.call_iii_tool("health", "analyze_wealth_acceleration_metrics", {"user_uuid": state['user_uuid']})

    @tool
    async def analyze_critical_survival_metrics_wrapper() -> str:
        """Analyze emergency funds and runway."""
        return await bridge.call_iii_tool("health", "analyze_critical_survival_metrics", {"user_uuid": state['user_uuid']})

    @tool
    async def plot_health_radar_wrapper() -> str:
        """Generate a visual financial health radar."""
        return await bridge.call_iii_tool("health", "plot_health_radar", {"user_uuid": state['user_uuid']})

    tools = [analyze_wealth_acceleration_metrics_wrapper, analyze_critical_survival_metrics_wrapper, plot_health_radar_wrapper]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a robotic data routing node for health. Think step-by-step."),
        ("human", "User Query: {input}\nUser ID: {user_uuid}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True, return_intermediate_steps=True)
    result = await agent_executor.ainvoke({"input": state['user_input'], "user_uuid": state['user_uuid']})
    output = str(result["intermediate_steps"][-1][1]) if "intermediate_steps" in result and result["intermediate_steps"] else result.get("output", "")
    cache_id, chart_type = None, None
    if output:
        match = re.search(r'\[TRIGGER_([A-Z_]+):([^\]]+)\]', output)
        if match:
            chart_type = match.group(1)
            cache_id = match.group(2).split(':')[0]
            output = re.sub(r'\[TRIGGER_[A-Z_]+:[^\]]+\]', '', output).strip()
    return {"worker_summary": output, "cache_id": cache_id, "chart_type": chart_type}
