import json
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

    logger.debug(json.dumps({"message": f"Core logger initialized for module: {module_name}", "status_code": 100}))

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
            logger.debug(json.dumps({"message": f"--- MCP REQUEST: [{tool_name}] ---", "status_code": 100}))
            logger.debug(json.dumps({"message": f"ARGS: {args}", "status_code": 100}))
            logger.debug(json.dumps({"message": f"KWARGS: {kwargs}", "status_code": 100}))

            try:
                result = func(*args, **kwargs)
                
                result_str = str(result)
                if len(result_str) > 1000:
                    result_str = result_str[:1000] + "... [TRUNCATED]"
                
                logger.debug(json.dumps({"message": f"--- MCP RESPONSE: [{tool_name}] ---", "status_code": 100}))
                logger.debug(json.dumps({"message": f"RESULT: {result_str}", "status_code": 100}))
                return result
            except Exception as e:
                logger.error(json.dumps({"message": f"--- MCP ERROR: [{tool_name}] ---", "status_code": 500}))
                logger.error(json.dumps({"message": f"EXCEPTION: {str(e)}", "status_code": 500}), exc_info=True)
                raise
        return wrapper
    return decorator
