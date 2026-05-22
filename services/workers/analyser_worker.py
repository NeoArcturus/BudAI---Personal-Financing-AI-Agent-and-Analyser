import re
import asyncio
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
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
            args=["-m", "mcp_servers.analyser_server"],
            env=env
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                yield session

async def run_analyser_worker(state):
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
    server_url = os.getenv("MCP_ANALYSER_URL")

    async with get_mcp_client(server_url) as session:
        await session.initialize()
        tools = await load_mcp_tools(session)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a robotic data routing node.
             AVAILABLE TOOLS: plot_expenses, find_total_spent_for_given_category, find_highest_spending_category, plot_cash_flow_mixed, export_custom_statement
             - Specific Category Totals: Use `find_total_spent_for_given_category` if they ask "how much did I spend on X".
             - Top Expenses: Use `find_highest_spending_category` to find out what they spent the most on.
             - Broad Spending/Expenses: Use `plot_expenses` to plot a line/bar chart of general spending over time.
             - Cash Flow/Income vs Expense: Use `plot_cash_flow_mixed` to show a mixed view of money in vs money out.
             - Download/Export: Use `export_custom_statement` to generate a downloadable CSV of their data.
             RULES:
             0. You MUST think step-by-step. Wrap your internal reasoning inside <think>...</think> tags before taking any action.
             1. You MUST use a tool to answer the user's query.
             2. Output ONLY the tool call. Do not explain.
             3. Once the tool returns data, return that exact data verbatim as your final answer. Do not add conversational filler.
             4. DEFAULT TO ALL: If the user does not type a specific bank name in their message, you MUST pass "ALL" as the bank_name_or_id parameter.
             5. SINGLE BANK: 'Plot my Wise expenses' -> You MUST pass "Wise".
             6. MULTIPLE BANKS: 'Chart my past expenses for Wise and Barclays' -> You MUST pass "Wise, Barclays".
             7. ACTIVE ACCOUNT OVERRIDE: ONLY use the "Active Account ID in UI" if the user explicitly types "this account" or "current account".
             """),
            ("human",
             "User Query: {input}\nUser ID: {user_uuid}\nActive Account ID in UI: {active_account_id}"),
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
            "user_uuid": state['user_uuid'],
            "active_account_id": state['active_account_id']
        })
        output = ""
        if "intermediate_steps" in result and len(result["intermediate_steps"]) > 0:
            action, observation = result["intermediate_steps"][-1]
            output = str(observation)
        else:
            output = result.get("output", "")
        cache_id = None
        chart_type = None
        if output:
            match = re.search(r'\[TRIGGER_([A-Z_]+):([^\]]+)\]', output)
            if match:
                chart_type = match.group(1)
                cache_id = match.group(2)
                output = re.sub(
                    r'\[TRIGGER_[A-Z_]+:[^\]]+\]', '', output).strip()
                logger.info(
                    f"Successfully extracted Cache ID: {cache_id}")
            else:
                logger.info(
                    f"No cache triggers found in tool output.")
        return {
            "worker_summary": output,
            "cache_id": cache_id,
            "chart_type": chart_type
        }
