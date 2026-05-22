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
from langchain_core.callbacks import BaseCallbackHandler
from contextlib import asynccontextmanager

logger = logging.getLogger("uvicorn.error")

class WorkerReasoningCallback(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs) -> None:
        try:
            content = response.generations[0][0].message.content
            if content:
                match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
                if match:
                    logger.info(f"Market Worker Reasoning:\n{match.group(1).strip()}")
                elif content.strip():
                    logger.info(f"Market Worker Thoughts:\n{content.strip()}")
        except Exception:
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
            args=["-m", "mcp_servers.macro_server"],
            env=env
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                yield session

async def run_market_worker(state):
    logger.info(f"Received market task: {state['user_input']}")

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
    server_url = os.getenv("MCP_MACRO_URL")

    async with get_mcp_client(server_url) as session:
        await session.initialize()
        tools = await load_mcp_tools(session)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Market Intelligence Agent.
             
             AVAILABLE TOOLS:
             - `get_live_market_data`: Fetches real-time price and change for specific tickers.
             - `get_financial_news`: Fetches latest financial news/headlines for a topic.

             RULES:
             0. You MUST think step-by-step using <think>...</think> tags.
             1. If the user asks about currency, stocks, or commodities, map it to the correct Yahoo Finance ticker (e.g., GBP to USD is 'GBPUSD=X', Gold is 'GC=F', Brent Crude is 'BZ=F', Apple is 'AAPL') and call get_live_market_data.
             2. If the user asks about the economy, inflation, or general financial news, call get_financial_news.
             3. Output ONLY the tool call. Do not explain.
             4. Once the tool returns data, return that exact data verbatim as your final answer. Do not add conversational filler.
             """),
            ("human", "User Query: {input}"),
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

        result = await agent_executor.ainvoke({"input": state['user_input']})

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
