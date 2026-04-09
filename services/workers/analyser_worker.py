import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama
from services.tools import plot_expenses, find_total_spent_for_given_category, find_highest_spending_category, plot_cash_flow_mixed


def run_analyser_worker(state):
    print(f"\n[ANALYSER WORKER] Received task: {state['user_input']}")

    llm = ChatOllama(
        model="qwen3:4b",
        temperature=0,
        keep_alive=300
    )

    tools = [plot_expenses, find_total_spent_for_given_category,
             find_highest_spending_category, plot_cash_flow_mixed]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a robotic data routing node.
         
         AVAILABLE TOOLS: plot_expenses, find_total_spent_for_given_category, find_highest_spending_category, plot_cash_flow_mixed

         - Specific Category Totals: Use `find_total_spent_for_given_category` if they ask "how much did I spend on X".
         - Top Expenses: Use `find_highest_spending_category` if they ask for their biggest drain or highest spend.
         - Past/Historical Trends: Use `plot_expenses` for daily/weekly/monthly historical spending trends. Do NOT use for forecasting. *CRITICAL: If the user asks for historical charts but does not specify a timeframe (daily/weekly/mothly), you MUST default to "Monthly" for plot_time_type parameter.*
         - Cash Flow: Use `plot_cash_flow_mixed` for income vs expense questions.
       
         
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

    print("\n[DEBUG] --- STARTING ANALYSER EXECUTION ---")

    result = agent_executor.invoke({
        "input": state['user_input'],
        "user_uuid": state['user_uuid'],
        "active_account_id": state['active_account_id']
    })

    print("[DEBUG] --- ENDING ANALYSER EXECUTION ---\n")

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
                f"[ANALYSER WORKER] Successfully extracted Cache ID: {cache_id}")
        else:
            print(f"[ANALYSER WORKER] No cache triggers found in tool output.")

    return {
        "worker_summary": output,
        "cache_id": cache_id,
        "chart_type": chart_type
    }
