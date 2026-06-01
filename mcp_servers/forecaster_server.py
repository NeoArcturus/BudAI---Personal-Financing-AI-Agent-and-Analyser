import warnings
import sys
import os
import asyncio
from iii import register_worker
from services.mcp_tools.forecaster_tools import generate_financial_forecast, generate_expense_forecast
from services.logger_setup import get_core_logger

warnings.filterwarnings("ignore")
logger = get_core_logger("forecaster_server")

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
    logger.info("Initializing Forecaster Worker with iii")
    worker = register_worker(III_ENGINE_URL)
    
    worker.register_function("forecaster::generate_financial_forecast", lambda args: tool_wrapper(generate_financial_forecast.func, args))
    worker.register_function("forecaster::generate_expense_forecast", lambda args: tool_wrapper(generate_expense_forecast.func, args))
    
    logger.info("Forecaster Worker registered. Entering heartbeat loop.")
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
