import os
import json
from typing import Annotated, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, InjectedState
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_uuid: str

@tool
def run_monte_carlo(
    state: Annotated[AgentState, InjectedState],
    days: int = 30,
    stress_test_active: bool = True,
    macro_environment: str = "Stable"
) -> str:
    """Runs Monte Carlo simulations (Bates stochastic jump-diffusion) to stress-test the user's financial runway under adverse economic conditions."""
    user_uuid = state.get("user_uuid")
    from services.analytics.forecast_service import compute_balance_forecast
    from services.api_integrator.account_reader import AccountReader
    reader = AccountReader(user_uuid)
    accounts = reader.get_all_accounts(skip_sync=True)
    if not accounts:
        return json.dumps({"status": "no_accounts", "message": "No active connected accounts found."})
    target_account = accounts[0]["account_id"]
    res = compute_balance_forecast(
        user_uuid=user_uuid,
        account_id=target_account,
        days=days,
        stress_test_active=stress_test_active,
        macro_environment=macro_environment
    )
    series_data = res.get("series", [{}])[0].get("data", [])
    if not series_data:
        return json.dumps({"status": "error", "message": "Simulation yielded no trajectory."})
    start_bal = series_data[0].get("Expected Balance", 0.0)
    end_careless = series_data[-1].get("careless_scenario", 0.0)
    end_expected = series_data[-1].get("Expected Balance", 0.0)
    end_optimal = series_data[-1].get("optimal_scenario", 0.0)
    ruin_days = [d["Day"] for d in series_data if d.get("careless_scenario", 0.0) <= 0.0]
    return json.dumps({
        "status": "success",
        "account_id": target_account,
        "starting_balance": start_bal,
        "days": days,
        "stress_test_active": stress_test_active,
        "macro_environment": macro_environment,
        "terminal_outcomes": {
            "expected": end_expected,
            "careless_95pct_risk": end_careless,
            "optimal": end_optimal
        },
        "critical_ruin_day": ruin_days[0] if ruin_days else None,
        "upcoming_scheduled_expenses_count": len(res.get("timeline", []))
    })

tools = [run_monte_carlo]

class MonteCarloSimulationAgent:
    """
    Standalone LangGraph Agent with ChatOpenAI capabilities for MonteCarloSimulationAgent.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        tool_node = ToolNode(tools)
        
        from services.llm_factory import get_chat_llm
        llm = get_chat_llm(temperature=0).bind_tools(tools)
        
        def evaluate_node(state: AgentState):
            logger.info("MonteCarloSimulationAgent evaluating state...")
            messages = list(state.get("messages", []))
            
            if not any(isinstance(m, SystemMessage) for m in messages):
                sys_msg = SystemMessage(
                    content=(
                        "### ROLE: Monte Carlo Simulation Specialist\n"
                        "You are the Monte Carlo Simulation Agent. Your job is to run thousands of background simulations to stress-test the user's financial runway under various economic conditions.\n\n"
                        "\n\n### STRICT ROLE BOUNDARIES & EXPERTISE ###\nYou are a highly specialized autonomous agent within the BudAI multi-agent network. Your role is strictly isolated to your specific domain of expertise. Do not attempt to execute actions outside your purview. You possess hyper-focused tools designed exclusively for your analytical tasks. Communicate professionally, mathematically, and directly. Avoid conversational filler.\n\n### STRICT ANTI-HALLUCINATION PROTOCOL ###\n"
                        "1. You MUST use your dedicated tools to perform actions or retrieve data.\n"
                        "2. Never hallucinate financial data or system states.\n"
                    )
                )
                messages.insert(0, sys_msg)
                
            response = llm.invoke(messages)
            return {"messages": [response]}
            
        def should_continue(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return END
            
        workflow = StateGraph(AgentState)
        workflow.add_node("evaluator", evaluate_node)
        workflow.add_node("tools", tool_node)
        
        workflow.set_entry_point("evaluator")
        workflow.add_conditional_edges("evaluator", should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "evaluator")
        
        return workflow.compile()
