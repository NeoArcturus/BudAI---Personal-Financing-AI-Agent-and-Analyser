import asyncio
import json
from typing import Optional, Dict, Any
from config import redis_client
from services.a2a.protocol import A2AMessage, A2AResponse
from services.a2a.registry import get_agent_registry
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class A2ABus:
    """
    Agent-to-Agent Communication Bus.
    Supports:
    - High-speed in-process direct dispatch via AgentRegistry
    - Distributed Redis Stream delivery (`a2a:mailbox:{agent_name}`)
    - Request-Response negotiation pattern with timeout handling
    """

    def __init__(self):
        self.registry = get_agent_registry()
        self.redis = redis_client

    async def send(self, msg: A2AMessage) -> None:
        """Asynchronously dispatches a message to the target agent mailbox."""
        if msg.recursion_depth > 5:
            raise ValueError(f"A2A Recursion ceiling exceeded (depth={msg.recursion_depth}) for correlation {msg.correlation_id}")

        # Try in-process handler first
        handler = self.registry.get_handler(msg.recipient)
        if handler:
            asyncio.create_task(self._invoke_handler(handler, msg))
            return

        # Fallback to Upstash Redis Stream
        if self.redis:
            stream_key = f"a2a:mailbox:{msg.recipient}"
            try:
                self.redis.xadd(stream_key, {"payload": msg.model_dump_json()})
            except Exception as e:
                logger.error(f"Failed to publish to Redis stream {stream_key}: {e}")

    async def request_reply(self, msg: A2AMessage) -> A2AResponse:
        """
        Synchronous request-response pattern. Dispatches the message and waits
        for the recipient's reply within msg.timeout_seconds.
        """
        if msg.recursion_depth > 5:
            return A2AResponse(
                reply_to_id=msg.message_id,
                correlation_id=msg.correlation_id,
                sender=msg.recipient,
                recipient=msg.sender,
                status="ERROR",
                error_message="Recursion depth limit exceeded"
            )

        handler = self.registry.get_handler(msg.recipient)
        if handler:
            try:
                return await asyncio.wait_for(
                    self._invoke_handler(handler, msg),
                    timeout=msg.timeout_seconds
                )
            except asyncio.TimeoutError:
                return A2AResponse(
                    reply_to_id=msg.message_id,
                    correlation_id=msg.correlation_id,
                    sender=msg.recipient,
                    recipient=msg.sender,
                    status="ERROR",
                    error_message=f"Timeout waiting for response from {msg.recipient}"
                )
            except Exception as e:
                return A2AResponse(
                    reply_to_id=msg.message_id,
                    correlation_id=msg.correlation_id,
                    sender=msg.recipient,
                    recipient=msg.sender,
                    status="ERROR",
                    error_message=str(e)
                )

        return A2AResponse(
            reply_to_id=msg.message_id,
            correlation_id=msg.correlation_id,
            sender=msg.recipient,
            recipient=msg.sender,
            status="ERROR",
            error_message=f"No registered handler for recipient agent '{msg.recipient}'"
        )

    async def _invoke_handler(self, handler, msg: A2AMessage) -> A2AResponse:
        if asyncio.iscoroutinefunction(handler):
            return await handler(msg)
        return handler(msg)

_bus_instance: Optional[A2ABus] = None

def get_a2a_bus() -> A2ABus:
    global _bus_instance
    if _bus_instance is None:
        _bus_instance = A2ABus()
    return _bus_instance
