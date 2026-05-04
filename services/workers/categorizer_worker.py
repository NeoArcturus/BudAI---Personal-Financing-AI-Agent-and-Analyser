import re
import asyncio
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("uvicorn.error")


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
            pass


async def run_categorizer_worker(state):
    logger.info(f"Received task: {state['user_input']}")

    llm = ChatOllama(
        model="qwen3:4b",
        temperature=0,
        keep_alive=300,
        callbacks=[WorkerReasoningCallback()]
    )

    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_servers.categorizer_server"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a robotic data routing node.
                 
                 AVAILABLE TOOLS: classify_financial_data, create_bargraph_chart_and_save

                 - Categorization & Breakdown: Use `classify_financial_data` whenever the user asks to categorize, classify, or break down their spending.
                 - Visual Category Charts: Use `create_bargraph_chart_and_save` if they explicitly want a bar chart or visual distribution of those categories.

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

            logger.info("--- STARTING CATEGORIZER EXECUTION ---")

            result = await agent_executor.ainvoke({
                "input": state['user_input'],
                "user_uuid": state['user_uuid'],
                "active_account_id": state['active_account_id']
            })

            logger.info("--- ENDING CATEGORIZER EXECUTION ---")

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
