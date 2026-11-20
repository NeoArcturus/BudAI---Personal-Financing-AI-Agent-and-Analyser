import os
import time
import httpx
import openai
import threading
from typing import Optional, Any, List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableConfig
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

DEFAULT_TAILNET_DOMAIN = os.getenv("TAILNET_DOMAIN", "tail49d1ce.ts.net")
DEFAULT_LLM_PORT = os.getenv("LLM_PORT", "1234")

# 4 Primary GPU inference desktops (RTX 4070 desktops via Tailscale MagicDNS)
DEFAULT_PRIMARY_NODES = [
    f"http://desktop-2cp9764.{DEFAULT_TAILNET_DOMAIN}:{DEFAULT_LLM_PORT}/v1",
    f"http://desktop-383je1s.{DEFAULT_TAILNET_DOMAIN}:{DEFAULT_LLM_PORT}/v1",
    f"http://desktop-t9nu4er.{DEFAULT_TAILNET_DOMAIN}:{DEFAULT_LLM_PORT}/v1",
    f"http://desktop-upmj9kg.{DEFAULT_TAILNET_DOMAIN}:{DEFAULT_LLM_PORT}/v1",
]

# 5th node: Standby / Failover backup Alienware laptop
DEFAULT_BACKUP_NODE = f"http://stpi-kohima-g11.{DEFAULT_TAILNET_DOMAIN}:{DEFAULT_LLM_PORT}/v1"

DEFAULT_MODEL_NAME = os.getenv("LLM_MODEL", "qwen3.5-9b")
DEFAULT_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "10.0"))

class ClusterNode:
    """Represents a single GPU inference machine on Tailscale."""
    def __init__(self, name: str, url: str, is_backup: bool = False):
        self.name = name
        self.url = url.rstrip("/")
        if not self.url.endswith("/v1"):
            self.url = f"{self.url}/v1"
        self.is_backup = is_backup
        self.active_requests: int = 0
        self.consecutive_failures: int = 0
        self.cooldown_until: float = 0.0

    def is_healthy(self, now: float) -> bool:
        return now >= self.cooldown_until

    def to_dict(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "name": self.name,
            "url": self.url,
            "is_backup": self.is_backup,
            "is_healthy": self.is_healthy(now),
            "active_requests": self.active_requests,
            "consecutive_failures": self.consecutive_failures,
            "cooldown_remaining_sec": max(0, int(self.cooldown_until - now))
        }

class LLMClusterManager:
    """
    In-process Client Load Balancer & Health Manager.
    - Least-Connections routing across the 4 primary RTX 4070 desktops.
    - Automatic tier-2 failover to the Alienware laptop when all 4 primaries are down.
    - Passive circuit breaker (30s cooldown on connection errors/timeouts).
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Parse primary nodes from env or defaults
        env_primaries = os.getenv("LLM_PRIMARY_ENDPOINTS")
        if env_primaries:
            primary_urls = [u.strip() for u in env_primaries.split(",") if u.strip()]
        else:
            primary_urls = DEFAULT_PRIMARY_NODES

        self.primary_nodes: List[ClusterNode] = []
        for url in primary_urls:
            name = url.split("//")[-1].split(".")[0].split(":")[0]
            self.primary_nodes.append(ClusterNode(name=name, url=url, is_backup=False))

        backup_url = os.getenv("LLM_BACKUP_ENDPOINT", DEFAULT_BACKUP_NODE)
        backup_name = backup_url.split("//")[-1].split(".")[0].split(":")[0]
        self.backup_node = ClusterNode(name=backup_name, url=backup_url, is_backup=True)

    @property
    def all_nodes(self) -> List[ClusterNode]:
        """Returns all distinct GPU nodes (4 primaries + 1 backup Alienware)."""
        seen = set()
        unique = []
        for n in self.primary_nodes + [self.backup_node]:
            if n.url not in seen:
                seen.add(n.url)
                unique.append(n)
        return unique

    def select_node(self, excluded_urls: Optional[set] = None) -> ClusterNode:
        """
        Selects the best healthy node:
        1. Filters healthy primary desktops (excluding already attempted nodes in the current request)
           and picks the one with fewest active requests.
        2. If all primaries are down or excluded, falls back to the Alienware backup node (if not excluded and healthy).
        3. If all nodes are exhausted or in cooldown, resets cooldown on the least-failing node and returns it.
        """
        now = time.time()
        excluded = excluded_urls or set()
        with self._lock:
            available_primaries = [n for n in self.primary_nodes if n.url not in excluded]
            healthy_primaries = [n for n in available_primaries if n.is_healthy(now)]
            if healthy_primaries:
                return min(healthy_primaries, key=lambda n: (n.active_requests, n.consecutive_failures))

            # Check backup Alienware node if not excluded and healthy
            if self.backup_node.url not in excluded and self.backup_node.is_healthy(now):
                logger.warning(
                    f"All available primary desktop nodes are down. "
                    f"Failing over to Alienware backup node: {self.backup_node.name} ({self.backup_node.url})"
                )
                return self.backup_node

            # If backup is also in cooldown or excluded, try any remaining primary that has not been tried yet
            if available_primaries:
                return min(available_primaries, key=lambda n: n.consecutive_failures)

            if self.backup_node.url not in excluded:
                return self.backup_node

            # Extreme edge case: all cluster nodes were already tried or in cooldown.
            # Reset the primary with the fewest failures
            logger.error("All cluster nodes exhausted or in cooldown. Resetting cooldown on least-failing node.")
            best_fallback = min(self.primary_nodes, key=lambda n: n.consecutive_failures)
            best_fallback.cooldown_until = 0.0
            return best_fallback

    def mark_success(self, url: str):
        with self._lock:
            for n in self.primary_nodes + [self.backup_node]:
                if n.url == url or f"{n.url}/" == url:
                    n.consecutive_failures = 0
                    n.cooldown_until = 0.0

    def mark_failure(self, url: str, error: Exception, base_cooldown: float = 60.0):
        now = time.time()
        with self._lock:
            for n in self.primary_nodes + [self.backup_node]:
                if n.url == url or f"{n.url}/" == url:
                    n.consecutive_failures += 1
                    # Exponential backoff capped at 300s (5 minutes): 60s, 120s, 240s, 300s
                    cooldown = min(300.0, base_cooldown * (2 ** min(n.consecutive_failures - 1, 4)))
                    n.cooldown_until = now + cooldown
                    logger.warning(
                        f"LLM Node [{n.name}] ({n.url}) marked DOWN for {cooldown:.1f}s "
                        f"(Failures: {n.consecutive_failures}). Error: {error}"
                    )

    def increment_active(self, url: str):
        with self._lock:
            for n in self.primary_nodes + [self.backup_node]:
                if n.url == url or f"{n.url}/" == url:
                    n.active_requests += 1

    def decrement_active(self, url: str):
        with self._lock:
            for n in self.primary_nodes + [self.backup_node]:
                if n.url == url or f"{n.url}/" == url:
                    n.active_requests = max(0, n.active_requests - 1)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_primary_nodes": len(self.primary_nodes),
                "primary_nodes": [n.to_dict() for n in self.primary_nodes],
                "backup_node": self.backup_node.to_dict()
            }

_cluster_manager_instance: Optional[LLMClusterManager] = None

def get_cluster_manager() -> LLMClusterManager:
    global _cluster_manager_instance
    if _cluster_manager_instance is None:
        _cluster_manager_instance = LLMClusterManager()
    return _cluster_manager_instance

RETRYABLE_ERROR_TYPES = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.APIStatusError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.HTTPStatusError,
)

class LoadBalancedChatOpenAI(ChatOpenAI):
    """
    ChatOpenAI wrapper that dynamically routes requests across the 4 primary
    RTX 4070 desktops via Least-Connections and auto-fails over to other desktop
    nodes and the Alienware backup laptop on network timeouts, connection drops,
    or node status errors (e.g. 400 Bad Request, 500/503 Service Unavailable).
    """

    def _sync_client_url(self, target_url: str):
        formatted_url = httpx.URL(f"{target_url}/" if not target_url.endswith("/") else target_url)
        if hasattr(self, "root_client") and self.root_client:
            self.root_client.base_url = formatted_url
        if hasattr(self, "root_async_client") and self.root_async_client:
            self.root_async_client.base_url = formatted_url
        if hasattr(self, "async_root_client") and self.async_root_client:
            self.async_root_client.base_url = formatted_url

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> ChatResult:
        mgr = get_cluster_manager()
        max_attempts = len(mgr.all_nodes)
        last_error = None
        attempted_urls = set()

        for attempt in range(max_attempts):
            node = mgr.select_node(excluded_urls=attempted_urls)
            attempted_urls.add(node.url)
            mgr.increment_active(node.url)
            self._sync_client_url(node.url)

            try:
                result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                mgr.mark_success(node.url)
                return result
            except RETRYABLE_ERROR_TYPES as err:
                last_error = err
                err_detail = getattr(err, "message", None) or getattr(err, "body", None) or str(err)
                mgr.mark_failure(node.url, err)
                logger.warning(
                    f"Inference attempt {attempt + 1}/{max_attempts} failed on {node.name} ({node.url}): {err_detail}; failing over..."
                )
            finally:
                mgr.decrement_active(node.url)

        if last_error:
            raise last_error
        raise RuntimeError("LLM cluster exhausted all retry attempts without a response.")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> ChatResult:
        mgr = get_cluster_manager()
        max_attempts = len(mgr.all_nodes)
        last_error = None
        attempted_urls = set()

        for attempt in range(max_attempts):
            node = mgr.select_node(excluded_urls=attempted_urls)
            attempted_urls.add(node.url)
            mgr.increment_active(node.url)
            self._sync_client_url(node.url)

            try:
                result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                mgr.mark_success(node.url)
                return result
            except RETRYABLE_ERROR_TYPES as err:
                last_error = err
                err_detail = getattr(err, "message", None) or getattr(err, "body", None) or str(err)
                mgr.mark_failure(node.url, err)
                logger.warning(
                    f"Async inference attempt {attempt + 1}/{max_attempts} failed on {node.name} ({node.url}): {err_detail}; failing over..."
                )
            finally:
                mgr.decrement_active(node.url)

        if last_error:
            raise last_error
        raise RuntimeError("LLM cluster exhausted all async retry attempts without a response.")

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ):
        mgr = get_cluster_manager()
        max_attempts = len(mgr.all_nodes)
        last_error = None
        attempted_urls = set()

        for attempt in range(max_attempts):
            node = mgr.select_node(excluded_urls=attempted_urls)
            attempted_urls.add(node.url)
            mgr.increment_active(node.url)
            self._sync_client_url(node.url)

            try:
                started = False
                async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
                    started = True
                    yield chunk
                mgr.mark_success(node.url)
                return
            except RETRYABLE_ERROR_TYPES as err:
                last_error = err
                err_detail = getattr(err, "message", None) or getattr(err, "body", None) or str(err)
                mgr.mark_failure(node.url, err)
                logger.warning(
                    f"Async stream attempt {attempt + 1}/{max_attempts} failed on {node.name} ({node.url}): {err_detail}; failing over..."
                )
                if started:
                    # Chunks were already yielded to consumer, cannot transparently switch node mid-stream
                    raise err
            finally:
                mgr.decrement_active(node.url)

        if last_error:
            raise last_error
        raise RuntimeError("LLM cluster exhausted all async stream retry attempts without a response.")

class LLMFactory:
    """
    Centralized factory for instantiating LangChain chat models with strict timeouts,
    consistent base URLs, zero external API keys, and automated 4-node GPU load balancing.
    """

    @staticmethod
    def get_chat_model(
        temperature: float = 0.0,
        timeout: Optional[float] = None,
        streaming: bool = False,
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any
    ) -> BaseChatModel:
        mgr = get_cluster_manager()
        initial_node = mgr.select_node()
        effective_base_url = base_url or initial_node.url
        effective_model = model_name or os.getenv("LLM_MODEL", DEFAULT_MODEL_NAME)
        effective_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT

        return LoadBalancedChatOpenAI(
            base_url=effective_base_url,
            api_key="budai-local",
            model=effective_model,
            temperature=temperature,
            timeout=effective_timeout,
            streaming=streaming,
            max_tokens=max_tokens,
            max_retries=1,
            **kwargs
        )

    @staticmethod
    def get_supervisor_model(
        temperature: float = 0.1,
        timeout: float = 60.0,
        streaming: bool = True,
        max_tokens: int = 4000,
        **kwargs: Any
    ) -> BaseChatModel:
        """Specialized model instance for the top-level supervisor/orchestrator graph."""
        return LLMFactory.get_chat_model(
            temperature=temperature,
            timeout=timeout,
            streaming=streaming,
            max_tokens=max_tokens,
            **kwargs
        )

def get_chat_llm(temperature: float = 0.0, timeout: Optional[float] = None, **kwargs: Any) -> BaseChatModel:
    return LLMFactory.get_chat_model(temperature=temperature, timeout=timeout, **kwargs)

def get_model_context_limit(model_name: Optional[str] = None) -> int:
    """Returns active model context window limit in tokens, defaulting to cluster limit (20224)."""
    env_limit = os.getenv("LLM_CONTEXT_LIMIT")
    if env_limit and env_limit.isdigit():
        return int(env_limit)
    return 20224

def get_cluster_status() -> Dict[str, Any]:
    return get_cluster_manager().get_status()
