import re
import asyncio
import logging
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from services.logger_setup import get_core_logger
from services.mcp_bridge import MCPBridge
from services.mcp_tools.market_tools import get_historical_market_data, compare_spending_to_market

logger = get_core_logger(__name__)

class WorkerReasoningCallback(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs) -> None:
        try:
            content = response.generations[0][0].message.content
            if content:
                match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
                if match: logger.info(f"Market Worker Reasoning: {match.group(1).strip()}")
                elif content.strip(): logger.info(f"Market Worker Thoughts: {content.strip()}")
        except Exception: pass

async def run_market_worker(state):
    logger.info(f"Received market task for user {state['user_uuid']}: {state['user_input']}")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
    if not base_url.endswith("/v1"): base_url = f"{base_url}/v1"
    llm = ChatOpenAI(model="mlx-community/Qwen2.5-7B-Instruct-4bit", base_url=base_url, api_key="budai-local", temperature=0, callbacks=[WorkerReasoningCallback()])
    bridge = MCPBridge()

    from services.profile_builder import ProfileBuilder
    profile_builder = ProfileBuilder(state['user_uuid'])
    mrfp = await profile_builder.build_profile()

    @tool
    async def get_live_market_data_wrapper(assets: list[str]) -> str:
        """Fetches real-time price and change for specific tickers."""
        return await bridge.call_iii_tool("macro", "get_live_market_data", {"assets": assets})

    @tool
    async def get_financial_news_wrapper(query: str) -> str:
        """Fetches latest financial news/headlines for a topic."""
        return await bridge.call_iii_tool("macro", "get_financial_news", {"query": query})

    @tool
    async def get_historical_market_data_wrapper(ticker: str, period: str = "6mo") -> str:
        """Fetches historical price trends and volatility for a specific ticker."""
        return get_historical_market_data.invoke({"ticker": ticker, "period": period})

    @tool
    async def compare_spending_to_market_wrapper(category: str, ticker: str) -> str:
        """Analyzes how a specific market asset (e.g. Oil, Gold) correlates with user spending in a category."""
        return compare_spending_to_market.invoke({"user_uuid": state['user_uuid'], "category": category, "ticker": ticker})

    tools = [
        get_live_market_data_wrapper, 
        get_financial_news_wrapper, 
        get_historical_market_data_wrapper, 
        compare_spending_to_market_wrapper
    ]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are the Market Intelligence Agent. 
         Your goal is to bridge the gap between global market trends and the user's personal financial state.

         ### USER PROFILE
         {mrfp}

         ROUTING RULES:
         1. For real-time prices/indices: use `get_live_market_data_wrapper`.
         2. For macroeconomic context (inflation, interest rates): use `get_financial_news_wrapper`.
         3. For analyzing personal impact (e.g. "How does gas price affect me?"): use `compare_spending_to_market_wrapper`.
         4. For historical trend analysis: use `get_historical_market_data_wrapper`.

         TICKER MAPPING:
         - Gold: 'GC=F', Silver: 'SI=F'
         - Oil (Brent): 'BZ=F', Oil (WTI): 'CL=F'
         - Natural Gas: 'NG=F'
         - S&P 500: '^GSPC', FTSE 100: '^FTSE'
         - GBP to USD: 'GBPUSD=X', GBP to EUR: 'GBPEUR=X'

         CRITICAL: Output ONLY the tool call. Once the tool returns, summarize the technical data in the context of the USER PROFILE above.
         """),
        ("human", "User Query: {input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True, return_intermediate_steps=True)
    
    result = await agent_executor.ainvoke({"input": state['user_input']})
    output = result.get("output", "Could not complete market analysis.")
    
    return {"worker_summary": output, "cache_id": None, "chart_type": None}
