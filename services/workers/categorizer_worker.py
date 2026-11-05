from services.workers.worker_utils import resolve_user_uuid
import re
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from models.graph_state import BudAIState
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

@tool("call_categorizer_agent", description="Use this tool to delegate the classification of raw merchant strings into verified taxonomy categories, handling unassigned transactions, and querying the semantic vector knowledge base for merchant data.")
async def call_categorizer_agent(
    query: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[BudAIState, InjectedState]
):
    """Delegate to the Categorizer Agent with automated A2A Analyser handoff for transaction fetching."""
    try:
        user_uuid = resolve_user_uuid(state)
        q_lower = query.lower()

        # If this is a general categorization or analysis query, execute via A2A bus with AnalyserAgent
        is_general_batch = any(kw in q_lower for kw in [
            "categorize", "classify", "uncategorized", "unassigned", "all", "highest", "spending", "breakdown"
        ])

        if is_general_batch:
            from services.a2a.bus import get_a2a_bus
            from services.a2a.protocol import A2AMessage
            bus = get_a2a_bus()
            a2a_msg = A2AMessage(
                sender="Supervisor",
                recipient="CategorizerAgent",
                message_type="COMMAND",
                intent="BATCH_CLASSIFY_UNCATEGORIZED",
                user_uuid=user_uuid,
                payload={"query": query, "include_breakdown": True, "limit": 40}
            )
            a2a_resp = await bus.request_reply(a2a_msg)
            if a2a_resp.status == "SUCCESS":
                p = a2a_resp.payload
                classified_count = p.get("classified_count", 0)
                merchants = p.get("merchants_classified", [])
                breakdown = p.get("category_breakdown") or {}

                lines = []
                if classified_count > 0:
                    lines.append(f"Successfully categorized {classified_count} unclassified merchant anchors across your connected accounts:")
                    for m in merchants[:8]:
                        lines.append(f"- {m['merchant']}: {m['category']} ({m['sub_category']}) [{m['transaction_count']} txs, £{m['total_amount']:.2f}]")
                    if len(merchants) > 8:
                        lines.append(f"- ...and {len(merchants) - 8} additional merchants.")
                else:
                    lines.append("All transactions in your connected accounts are already categorized.")

                cats = breakdown.get("categories", [])
                if cats:
                    lines.append("\nCurrent Spending Breakdown by Category:")
                    for c in cats[:8]:
                        lines.append(f"- {c['category']}: £{c['total_spent']:.2f} ({c['transaction_count']} transactions)")

                highest = breakdown.get("highest_spending_per_category", [])
                if highest:
                    lines.append("\nHighest Spending Transactions by Category:")
                    for h in highest[:6]:
                        lines.append(f"- {h['category']}: £{h['amount']:.2f} at {h['description']} ({h['date'] or 'Recent'})")

                output_text = "\n".join(lines)
                return Command(update={
                    "messages": [
                        ToolMessage(content=output_text, tool_call_id=tool_call_id)
                    ]
                })

        # Fallback to direct LangGraph model invocation for single merchant queries
        from agents.core_financial.Categorizer_Agent.CategorizerAgent import CategorizerAgent
        agent = CategorizerAgent()
        
        result = await agent.app.ainvoke({"messages": [{"role": "user", "content": query}], "user_uuid": user_uuid})
        raw_output = result["messages"][-1].content
        output = raw_output if isinstance(raw_output, str) else "".join([b if isinstance(
            b, str) else b.get("text", "") for b in raw_output if isinstance(b, (str, dict))])
        
        cache_id, chart_type = None, None
        match = re.search(r'\[TRIGGER_([A-Z_]+):([^\]]+)\]', output)
        if match:
            chart_type = match.group(1)
            cache_id = match.group(2).split(':')[0]
            output = re.sub(r'\[TRIGGER_[A-Z_]+:[^\]]+\]', '', output).strip()

        return Command(update={
            "cache_id": cache_id,
            "chart_type": chart_type,
            "messages": [
                ToolMessage(content=output, tool_call_id=tool_call_id)
            ]
        })
    except Exception as e:
        logger.error(f"Error in categorizer worker: {e}")
        return Command(update={
            "messages": [
                ToolMessage(content=f"Error executing Categorizer tool: {str(e)}", tool_call_id=tool_call_id)
            ]
        })
