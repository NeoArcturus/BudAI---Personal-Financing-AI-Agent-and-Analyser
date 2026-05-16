import asyncio
import logging
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from services.mcp_tools.external_tools import search_user_memory, save_to_user_memory
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


async def run_memory_worker(state):
    logger.info(f"Received task: {state['user_input']}")

    llm = ChatOllama(
        model="qwen3:4b",
        temperature=0,
        keep_alive=300,
        callbacks=[WorkerReasoningCallback()]
    )

    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_servers.memory_server"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await load_mcp_tools(session)
            
            # Combine static memory tools and dynamic MCP history tools
            tools = mcp_tools + [search_user_memory, save_to_user_memory]

            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are the Knowledge & History Agent.
                 
                 AVAILABLE TOOLS:
                 - `save_to_user_memory`: Save specific facts, goals, or preferences the user states.
                 - `search_user_memory`: Look up previously saved facts or preferences.
                 - `search_financial_history_semantic`: Perform a semantic search on the user's transaction history to find patterns, trends, or specific historical spending categories (e.g. "travel habits", "luxury spending").

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

            logger.info("--- STARTING MEMORY EXECUTION ---")
            result = await agent_executor.ainvoke({
                "input": state['user_input'],
                "user_uuid": state['user_uuid']
            })
            logger.info("--- ENDING MEMORY EXECUTION ---")

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
