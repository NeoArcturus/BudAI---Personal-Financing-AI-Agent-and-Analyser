import asyncio
import logging
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama
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

    tools = [search_user_memory, save_to_user_memory]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the Knowledge Graph administrator.\nAVAILABLE TOOLS:\n- `save_to_user_memory`: Use this to save facts, preferences, constraints, or goals the user tells you to remember.\n- `search_user_memory`: Use this to look up stored facts or preferences if the user asks what you know.\nRULES:\n0. You MUST think step-by-step. Wrap your internal reasoning inside <think>...</think> tags before taking any action.\n1. You MUST use a tool to interact with the memory graph.\n2. When saving, extract the core entities.\n3. Output ONLY the tool call. Do not explain.\n4. Once the tool returns data, summarize what was saved or found as your final answer."),
        ("human", "User Query: {input}\nUser ID: {user_uuid}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

    logger.info("--- STARTING MEMORY EXECUTION ---")
    try:
        result = await agent_executor.ainvoke({
            "input": state['user_input'],
            "user_uuid": state['user_uuid']
        })
        output = result.get("output", "Memory operation completed.")
    except Exception as e:
        logger.error(f"Execution Failed: {e}")
        output = "Failed to interact with the Knowledge Graph."
    logger.info("--- ENDING MEMORY EXECUTION ---")

    return {
        "worker_summary": output,
        "cache_id": None,
        "chart_type": None
    }
