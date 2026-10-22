import json
import logging
import sys
import time
import functools
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

class StructuredJsonFormatter(logging.Formatter):
    """
    Formats log records as structured, single-line JSON objects routed to sys.stderr.
    Compatible with Google Cloud Logging, Datadog, Loki, and local dev environments.
    Seamlessly extracts data payloads and status codes while unpacking legacy JSON-in-string logs.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
        }

        # Attempt to parse legacy JSON-in-string payloads or direct dict messages
        raw_msg = record.msg
        parsed_from_msg = False
        if isinstance(raw_msg, dict):
            msg_dict = raw_msg
            parsed_from_msg = True
        elif isinstance(raw_msg, str) and raw_msg.strip().startswith("{") and raw_msg.strip().endswith("}"):
            try:
                msg_dict = json.loads(raw_msg)
                if isinstance(msg_dict, dict):
                    parsed_from_msg = True
                else:
                    msg_dict = {}
            except Exception:
                parsed_from_msg = False
                msg_dict = {}
        else:
            msg_dict = {}

        if parsed_from_msg:
            log_payload["message"] = str(msg_dict.get("message", record.getMessage()))
            if "status_code" in msg_dict:
                log_payload["status_code"] = msg_dict["status_code"]
            if "data" in msg_dict:
                log_payload["data"] = msg_dict["data"]
            # Elevate any other contextual keys
            for k, v in msg_dict.items():
                if k not in ["message", "status_code", "data"]:
                    log_payload[k] = v
        else:
            log_payload["message"] = record.getMessage()

        # Capture extra fields passed via extra={"data": ..., "status_code": ...}
        standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "taskName", "message"
        }
        for key, val in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                log_payload[key] = val

        # Capture exception tracebacks cleanly in a dedicated field
        if record.exc_info:
            log_payload["exception"] = "".join(traceback.format_exception(*record.exc_info)).strip()

        return json.dumps(log_payload, default=str)


def get_core_logger(module_name: str) -> logging.Logger:
    """
    Returns a configured hierarchical logger strictly writing to sys.stderr.
    Prevents corrupting MCP JSON-RPC streams on stdout while ensuring high-signal logs.
    """
    logger_name = f"budai.{module_name}" if not module_name.startswith("budai.") else module_name
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(console_handler)
        logger.propagate = False

    return logger


def log_pipeline_data(logger: logging.Logger, pipeline: str, stage: str, data: Any, message: Optional[str] = None):
    """
    Dedicated utility to display exact data content flowing through core pipelines
    (e.g. Ingestion, Delta Sync, Categorization, Subscriptions, Priority Cascade).
    """
    msg = message or f"Pipeline [{pipeline}] -> Stage [{stage}]"
    logger.info(msg, extra={
        "pipeline": pipeline,
        "stage": stage,
        "data": data
    })


def log_mcp_tool(logger: logging.Logger):
    """
    Decorator to log MCP tool execution and data flow (inputs and outputs) cleanly to stderr.
    Consolidates requests and responses into structured data events.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tool_name = func.__name__
            start_time = time.time()

            # Sanitize and summarize arguments to display data flow
            args_clean = [str(a)[:500] for a in args]
            kwargs_clean = {k: (str(v)[:500] if len(str(v)) > 500 else v) for k, v in kwargs.items()}

            logger.info(
                f"MCP Tool Request: [{tool_name}]",
                extra={
                    "mcp_tool": tool_name,
                    "stage": "request",
                    "data": {"args": args_clean, "kwargs": kwargs_clean}
                }
            )

            try:
                result = func(*args, **kwargs)
                elapsed_ms = int((time.time() - start_time) * 1000)
                
                result_repr = str(result)
                if len(result_repr) > 2000:
                    result_repr = result_repr[:2000] + "... [TRUNCATED]"

                logger.info(
                    f"MCP Tool Response: [{tool_name}] ({elapsed_ms}ms)",
                    extra={
                        "mcp_tool": tool_name,
                        "stage": "response",
                        "duration_ms": elapsed_ms,
                        "data": {"result": result_repr}
                    }
                )
                return result
            except Exception as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                logger.error(
                    f"MCP Tool Error: [{tool_name}] ({elapsed_ms}ms): {str(e)}",
                    extra={
                        "mcp_tool": tool_name,
                        "stage": "error",
                        "duration_ms": elapsed_ms,
                        "data": {"error": str(e)}
                    },
                    exc_info=True
                )
                raise
        return wrapper
    return decorator
