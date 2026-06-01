import logging
import sys


def get_core_logger(module_name: str) -> logging.Logger:
    logger = logging.getLogger("uvicorn.error")

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.propagate = False

    logger.debug(f"Core logger initialized for module: {module_name}")

    return logger


def log_mcp_tool(logger: logging.Logger):
    """
    Decorator to log MCP tool requests and responses to stderr.
    """
    def decorator(func):
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tool_name = func.__name__
            logger.info(f"--- MCP REQUEST: [{tool_name}] ---")
            logger.info(f"ARGS: {args}")
            logger.info(f"KWARGS: {kwargs}")

            try:
                result = func(*args, **kwargs)
                
                # Truncate large output for cleaner logs
                result_str = str(result)
                if len(result_str) > 1000:
                    result_str = result_str[:1000] + "... [TRUNCATED]"
                
                logger.info(f"--- MCP RESPONSE: [{tool_name}] ---")
                logger.info(f"RESULT: {result_str}")
                return result
            except Exception as e:
                logger.error(f"--- MCP ERROR: [{tool_name}] ---")
                logger.error(f"EXCEPTION: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator
