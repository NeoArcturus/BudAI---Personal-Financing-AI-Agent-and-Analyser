import warnings
import sys
import os
import asyncio
from iii import register_worker
from services.mcp_tools.categorizer_tools import classify_financial_data, create_bargraph_chart_and_save, create_pie_chart_and_save, update_transaction_category, retrain_categorization_model
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
    logger.info("Initializing Categorizer Worker with iii")
    worker = register_worker(III_ENGINE_URL)
    
    worker.register_function("categorizer::classify_financial_data", lambda args: tool_wrapper(classify_financial_data.func, args))
    worker.register_function("categorizer::create_bargraph_chart_and_save", lambda args: tool_wrapper(create_bargraph_chart_and_save.func, args))
    worker.register_function("categorizer::create_pie_chart_and_save", lambda args: tool_wrapper(create_pie_chart_and_save.func, args))
    worker.register_function("categorizer::update_transaction_category", lambda args: tool_wrapper(update_transaction_category.func, args))
    worker.register_function("categorizer::retrain_categorization_model", lambda args: tool_wrapper(retrain_categorization_model.func, args))
    
    logger.info("Categorizer Worker registered. Entering heartbeat loop.")
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
