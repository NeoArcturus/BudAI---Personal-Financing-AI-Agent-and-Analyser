import os
import json
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from services.orchestration.execution_tools import get_sleep_tool, get_alert_tool, get_transfer_tool, get_etl_tool, get_create_bucket_tool, get_update_bucket_tool, get_delete_bucket_tool
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class ExecutionState(TypedDict):
    messages: Sequence[BaseMessage]

def execute_llm_chain(user_uuid: str, compiled_brief: str, reverse_map: dict):
    """
    Step 1 & 2 of Phase 4: The LLM Decision Engine Execution
    Initializes a highly scoped LangGraph state machine, binds the mathematically verified tools, 
    and forces the LLM to make a decision based exclusively on the state snapshot.
    """
    logger.info(json.dumps({"message": f"Waking Decision Engine for {user_uuid}", "status_code": 200}))
    
    # Generate tightly bound tools using closures to protect PII/UUIDs
    tools = [
        get_sleep_tool(),
        get_alert_tool(user_uuid),
        get_transfer_tool(reverse_map, user_uuid),
        get_etl_tool(),
        get_create_bucket_tool(user_uuid),
        get_update_bucket_tool(reverse_map, user_uuid),
        get_delete_bucket_tool(reverse_map, user_uuid)
    ]
    tool_node = ToolNode(tools)
    
    # Configure LLM
    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:8000/v1")
    if not base_url.endswith("/v1"): 
        base_url = f"{base_url}/v1"
        
    llm = ChatOpenAI(
        base_url=base_url,
        api_key="budai-local",
        model="Qwen3.5-9B-GGUF",
        temperature=0
    )
    llm_with_tools = llm.bind_tools(tools)
    
    def evaluate_node(state: ExecutionState):
        logger.info(json.dumps({"message": "Decision Engine evaluating state.", "status_code": 200}))
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
        
    def should_continue(state: ExecutionState):
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        return "continue"
        
    workflow = StateGraph(ExecutionState)
    workflow.add_node("evaluator", evaluate_node)
    workflow.add_node("tools", tool_node)
    
    workflow.set_entry_point("evaluator")
    workflow.add_conditional_edges(
        "evaluator",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    # Once the tool fires (e.g. money moved or alert sent), the loop terminates
    workflow.add_edge("tools", END) 
    
    engine = workflow.compile()
    
    sys_prompt = (
        "You are BudAI, an autonomous, zero-sum financial allocator. Your job is to manage the user's virtual buckets proactively.\n"
        "You operate in a strict zero-sum environment. You cannot create money. To fund a bucket, you must transfer it from another bucket (usually DEFAULT).\n"
        "You have been provided with native tools (create bucket, transfer money, alert user, etc.).\n"
        "Do NOT output plain text JSON commands. You MUST invoke your tools directly to take action. If the financial state requires no action, invoke the sleep tool."
    )
    
    state = {
        "messages": [SystemMessage(content=sys_prompt), HumanMessage(content=compiled_brief)]
    }
    
    try:
        engine.invoke(state)
        logger.info(json.dumps({"message": "Decision Engine completed execution.", "status_code": 200}))
    except Exception as e:
        logger.error(json.dumps({"message": f"Decision Engine failed: {e}", "status_code": 500}))
