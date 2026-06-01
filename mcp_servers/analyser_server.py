import warnings
import sys
import os
import asyncio
from iii import register_worker
from services.mcp_tools.analyser_tools import plot_expenses, find_total_spent_for_given_category, find_highest_spending_category, plot_cash_flow_mixed
from services.mcp_tools.external_tools import export_custom_statement
from services.logger_setup import get_core_logger

warnings.filterwarnings("ignore")
logger = get_core_logger(__name__)

III_ENGINE_URL = os.getenv("III_ENGINE_URL", "ws://iii-engine:49134")

def tool_wrapper(func, args):
    # Filter out internal iii arguments
    clean_args = {k: v for k, v in args.items() if not k.startswith("_")}
    try:
        if asyncio.iscoroutinefunction(func):
            try:
                return asyncio.run(func(**clean_args))
            except RuntimeError:
                # Fallback if loop is already running
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(func(**clean_args))
        else:
            return func(**clean_args)
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {e}", exc_info=True)
        return f"Error: {str(e)}"

async def main():
    logger.info("Initializing Analyser Worker with iii")
    worker = register_worker(III_ENGINE_URL)
    
    worker.register_function("analyser::plot_expenses", lambda args: tool_wrapper(plot_expenses.func, args))
    worker.register_function("analyser::find_total_spent_for_given_category", lambda args: tool_wrapper(find_total_spent_for_given_category.func, args))
    worker.register_function("analyser::find_highest_spending_category", lambda args: tool_wrapper(find_highest_spending_category.func, args))
    worker.register_function("analyser::plot_cash_flow_mixed", lambda args: tool_wrapper(plot_cash_flow_mixed.func, args))
    worker.register_function("analyser::export_custom_statement", lambda args: tool_wrapper(export_custom_statement.func, args))
    
    logger.info("Analyser Worker registered. Entering heartbeat loop.")
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
