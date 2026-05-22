import asyncio
import logging
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from services.mcp_tools.external_tools import search_user_memory, save_to_user_memory
from langchain_core.callbacks import BaseCallbackHandler
from contextlib import asynccontextmanager
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class WorkerReasoningCallback(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs) -> None:
        try:
            content = response.generations[0][0].message.content
            if content:
                match = re.search(r"<think>(.*?)</think>",
                                  content, flags=re.DOTALL)
                if match:
                    logger.info(
                        f"Worker Reasoning:\n{match.group(1).strip()}")
                elif content.strip():
                    logger.info(f"Worker Thoughts:\n{content.strip()}")
        except Exception:
            logger.error("An error occurred in this block", exc_info=True)
            pass

@asynccontextmanager
async def get_mcp_client(server_url: str = None):
    import os
    if server_url:
        logger.info(f"Connecting to MCP server via SSE: {server_url}")
        async with sse_client(server_url) as (read, write):
            async with ClientSession(read, write) as session:
                yield session
    else:
        logger.info("Spawning local MCP server via stdio")
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore"
        server_params = StdioServerParameters(
            command="python",
            args=["-m", "mcp_servers.memory_server"],
            env=env
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                yield session

async def run_memory_worker(state):
    logger.info(f"Received task: {state['user_input']}")
    import os
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = ChatOllama(
        model="qwen3:4b",
        base_url=base_url,
        temperature=0,
        keep_alive=300,
        callbacks=[WorkerReasoningCallback()]
    )
    
    import os
    server_url = os.getenv("MCP_MEMORY_URL")

    async with get_mcp_client(server_url) as session:
        await session.initialize()
        mcp_tools = await load_mcp_tools(session)
        tools = mcp_tools + [search_user_memory, save_to_user_memory]
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Knowledge & History Agent.
             AVAILABLE TOOLS:
             - `save_to_user_memory`: Save specific facts, goals, or preferences the user states.
             - `search_user_memory`: Look up previously saved facts or preferences.
             - `search_financial_history_semantic`: Perform a semantic search on the user's transaction history to find patterns, trends, or specific historical spending categories (e.g. "travel habits", "luxury spending").
             - `get_seasonal_behavior_context`: Retrieves a summary of how the user historically behaves in the current month.
             RULES:
             0. You MUST think step-by-step. Wrap your internal reasoning inside <think>...</think> tags before taking any action.
             1. You MUST use a tool to answer the user's query.
             2. Output ONLY the tool call. Do not explain.
             3. Once the tool returns data, return that exact data verbatim as your final answer. Do not add conversational filler.
             """),
            ("human",
             "User Query: {input}\nUser ID: {user_uuid}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        result = await agent_executor.ainvoke({
            "input": state['user_input'],
            "user_uuid": state['user_uuid']
        })
        output = ""
        if "intermediate_steps" in result and len(result["intermediate_steps"]) > 0:
            action, observation = result["intermediate_steps"][-1]
            output = str(observation)
        else:
            output = result.get("output", "")
        return {
            "worker_summary": output,
            "cache_id": None,
            "chart_type": None
        }
