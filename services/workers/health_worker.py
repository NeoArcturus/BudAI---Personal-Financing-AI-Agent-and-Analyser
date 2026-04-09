import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama
from services.tools import analyze_wealth_acceleration_metrics, analyze_critical_survival_metrics, plot_health_radar


def run_health_worker(state):
    print(f"\n[HEALTH WORKER] Received task: {state['user_input']}")

    llm = ChatOllama(
        model="qwen3:4b",
        temperature=0,
        keep_alive=300
    )

    tools = [analyze_wealth_acceleration_metrics,
             analyze_critical_survival_metrics, plot_health_radar]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a robotic data routing node.
         
         AVAILABLE TOOLS: analyze_wealth_acceleration_metrics,
             analyze_critical_survival_metrics, plot_health_radar
         
         - Wealth/Health: Use `analyze_wealth_acceleration_metrics` or `plot_health_radar` for general financial health.
         - Survival/Debt: Use `analyze_critical_survival_metrics` for emergency funds, runway, or debt repayment questions. 
         
         RULES:
         1. You MUST use a tool to answer the user's query.
         2. Output ONLY the tool call. Do not explain.
         3. Once the tool returns data, return that exact data verbatim as your final answer. Do not add conversational filler.
         4. ACCOUNT SELECTION RULES (CRITICAL)
            - DEFAULT TO ALL: If the user does not type a specific bank name in their message, you MUST pass "ALL" as the `bank_name_or_id` parameter.
            - SINGLE BANK: "Plot my Wise expenses" -> You MUST pass "Wise".
            - MULTIPLE BANKS: "Chart my past expenses for Wise and Barclays" -> You MUST pass "Wise, Barclays" as a single comma-separated string.
            - ACTIVE ACCOUNT OVERRIDE: ONLY use the 'Active Account ID in UI' if the user explicitly types the exact words "this account" or "current account".
         """),
        ("human",
         "User Query: {input}\nUser ID: {user_uuid}\nAccount: {active_account_id}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    print("\n[DEBUG] --- STARTING HEALTH EXECUTION ---")

    result = agent_executor.invoke({
        "input": state['user_input'],
        "user_uuid": state['user_uuid'],
        "active_account_id": state['active_account_id']
    })

    print("[DEBUG] --- ENDING HEALTH EXECUTION ---\n")

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
            output = re.sub(r'\[TRIGGER_[A-Z_]+:[^\]]+\]', '', output).strip()
            print(
                f"[HEALTH WORKER] Successfully extracted Cache ID: {cache_id}")
        else:
            print(f"[HEALTH WORKER] No cache triggers found in tool output.")

    return {
        "worker_summary": output,
        "cache_id": cache_id,
        "chart_type": chart_type
    }
