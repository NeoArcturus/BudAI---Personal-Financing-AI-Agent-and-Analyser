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

# 4 Primary RTX 4070 desktop nodes (LM Studio via Tailscale MagicDNS)
DEFAULT_PRIMARY_NODES = [
    f"http://desktop-2cp9764.{DEFAULT_TAILNET_DOMAIN}:{DEFAULT_LLM_PORT}/v1",
    f"http://desktop-383je1s.{DEFAULT_TAILNET_DOMAIN}:{DEFAULT_LLM_PORT}/v1",
    f"http://desktop-t9nu4er.{DEFAULT_TAILNET_DOMAIN}:{DEFAULT_LLM_PORT}/v1",
    f"http://desktop-upmj9kg.{DEFAULT_TAILNET_DOMAIN}:{DEFAULT_LLM_PORT}/v1",
]

# Backup Alienware laptop node (activated strictly when all 4 primary desktops fail)
DEFAULT_BACKUP_NODE = f"http://stpi-kohima-g11.{DEFAULT_TAILNET_DOMAIN}:{DEFAULT_LLM_PORT}/v1"

DEFAULT_MODEL_NAME = os.getenv("LLM_MODEL", "qwen3.5-9b")
DEFAULT_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30.0"))

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
        elif os.getenv("OLLAMA_BASE_URL"):
            # If explicit single URL provided (e.g. local offline dev)
            primary_urls = [os.getenv("OLLAMA_BASE_URL")]
        else:
            primary_urls = DEFAULT_PRIMARY_NODES

        self.primary_nodes: List[ClusterNode] = []
        for url in primary_urls:
            name = url.split("//")[-1].split(".")[0].split(":")[0]
            self.primary_nodes.append(ClusterNode(name=name, url=url, is_backup=False))

        backup_url = os.getenv("LLM_BACKUP_ENDPOINT", DEFAULT_BACKUP_NODE)
        backup_name = backup_url.split("//")[-1].split(".")[0].split(":")[0]
        self.backup_node = ClusterNode(name=backup_name, url=backup_url, is_backup=True)

    def select_node(self) -> ClusterNode:
        """
        Selects the best healthy node:
        1. Filters healthy primary desktops and picks the one with fewest active requests.
        2. If all 4 primaries are in cooldown, falls back to the Alienware backup node.
        3. If all nodes including backup are down, resets cooldowns and retries least-failing desktop.
        """
        now = time.time()
        with self._lock:
            healthy_primaries = [n for n in self.primary_nodes if n.is_healthy(now)]
            if healthy_primaries:
                # Least-Connections routing
                return min(healthy_primaries, key=lambda n: n.active_requests)

            # All 4 primary desktops are down; check backup Alienware node
            if self.backup_node.is_healthy(now):
                logger.warning(
                    f"All {len(self.primary_nodes)} primary desktop nodes are down. "
                    f"Failing over to Alienware backup node: {self.backup_node.name} ({self.backup_node.url})"
                )
                return self.backup_node

            # Extreme edge case: all primaries and backup are down. Reset the least-failing primary.
            logger.error("All primary nodes and backup node are in cooldown. Resetting cooldown on primary pool.")
            best_fallback = min(self.primary_nodes, key=lambda n: n.consecutive_failures)
            best_fallback.cooldown_until = 0.0
            return best_fallback

    def mark_success(self, url: str):
        with self._lock:
            for n in self.primary_nodes + [self.backup_node]:
                if n.url == url or f"{n.url}/" == url:
                    n.consecutive_failures = 0
                    n.cooldown_until = 0.0

    def mark_failure(self, url: str, error: Exception, cooldown_seconds: float = 30.0):
        now = time.time()
        with self._lock:
            for n in self.primary_nodes + [self.backup_node]:
                if n.url == url or f"{n.url}/" == url:
                    n.consecutive_failures += 1
                    n.cooldown_until = now + cooldown_seconds
                    logger.warning(
                        f"LLM Node [{n.name}] ({n.url}) marked DOWN for {cooldown_seconds}s "
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

NETWORK_ERROR_TYPES = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
)

class LoadBalancedChatOpenAI(ChatOpenAI):
    """
    ChatOpenAI wrapper that dynamically routes requests across the 4 primary
    RTX 4070 desktops via Least-Connections and auto-fails over to the Alienware
    backup laptop on network timeouts or connection drops.
    """

    def _sync_client_url(self, target_url: str):
        formatted_url = httpx.URL(f"{target_url}/" if not target_url.endswith("/") else target_url)
        self.root_client.base_url = formatted_url
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
        max_attempts = len(mgr.primary_nodes) + 1
        last_error = None

        for attempt in range(max_attempts):
            node = mgr.select_node()
            mgr.increment_active(node.url)
            self._sync_client_url(node.url)

            try:
                result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                mgr.mark_success(node.url)
                return result
            except NETWORK_ERROR_TYPES as err:
                last_error = err
                mgr.mark_failure(node.url, err)
                logger.warning(f"Inference attempt {attempt + 1}/{max_attempts} failed on {node.name}; failing over...")
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
        max_attempts = len(mgr.primary_nodes) + 1
        last_error = None

        for attempt in range(max_attempts):
            node = mgr.select_node()
            mgr.increment_active(node.url)
            self._sync_client_url(node.url)

            try:
                result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                mgr.mark_success(node.url)
                return result
            except NETWORK_ERROR_TYPES as err:
                last_error = err
                mgr.mark_failure(node.url, err)
                logger.warning(f"Async inference attempt {attempt + 1}/{max_attempts} failed on {node.name}; failing over...")
            finally:
                mgr.decrement_active(node.url)

        if last_error:
            raise last_error
        raise RuntimeError("LLM cluster exhausted all async retry attempts without a response.")

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

def get_cluster_status() -> Dict[str, Any]:
    return get_cluster_manager().get_status()
