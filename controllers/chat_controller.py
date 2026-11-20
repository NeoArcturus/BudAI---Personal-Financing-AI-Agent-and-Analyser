import sys
import asyncio
import uuid
import json
import time
from typing import List
from fastapi import HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from langfuse.langchain import CallbackHandler
from langgraph.types import Command
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

from models.database_models import User, ChatSession, ChatHistory
from schemas.api_schema import (
    ChatRequest, StreamChatRequest, ChatSessionResponse,
    ChatMessageResponse, ChatSessionRenameRequest
)
from services.orchestrator_graph import execute_chat_graph_async, get_session_history
from agents.finance.Analyser_Agent.AnalyserAgent import get_analyser_app
from config import SessionLocal, redis_client
from services.llm_manager import GlobalLLMManager
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

budai_app = get_analyser_app()


async def get_task_status(task_id: str):
    """Retrieves the status of an asynchronous chat task from Redis."""
    task_info_raw = redis_client.get(f"task:{task_id}")
    if not task_info_raw:
        raise HTTPException(status_code=404, detail="Task not found")
    return json.loads(task_info_raw)


async def list_chat_sessions(current_user: User) -> List[ChatSessionResponse]:
    """Lists all chat sessions for the current user."""
    try:
        with SessionLocal() as session:
            sessions = session.query(ChatSession).filter_by(
                user_uuid=current_user.user_uuid
            ).order_by(ChatSession.last_updated.desc()).all()
            return [
                ChatSessionResponse(
                    session_id=s.session_id,
                    title=s.title or "New Conversation",
                    last_updated=s.last_updated,
                    context_data=s.context_data
                ) for s in sessions
            ]
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history.")


async def get_chat_session(session_id: str, current_user: User) -> ChatSessionResponse:
    """Retrieves a specific chat session and message history."""
    try:
        with SessionLocal() as session:
            chat_session = session.query(ChatSession).filter_by(
                session_id=session_id, user_uuid=current_user.user_uuid
            ).first()
            if not chat_session:
                raise HTTPException(status_code=404, detail="Session not found.")
            messages = session.query(ChatHistory).filter_by(
                session_id=session_id
            ).order_by(ChatHistory.timestamp.asc()).all()

            message_responses = [
                ChatMessageResponse(
                    role=m.role,
                    content=m.content,
                    reasoning_content=m.reasoning_content,
                    timestamp=m.timestamp
                ) for m in messages
            ]

            return ChatSessionResponse(
                session_id=chat_session.session_id,
                title=chat_session.title or "New Conversation",
                last_updated=chat_session.last_updated,
                context_data=chat_session.context_data,
                messages=message_responses
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to retrieve session details.")


async def rename_chat_session(session_id: str, request: ChatSessionRenameRequest, current_user: User):
    """Renames the title of a specific chat session."""
    try:
        with SessionLocal() as session:
            chat_session = session.query(ChatSession).filter_by(
                session_id=session_id, user_uuid=current_user.user_uuid
            ).first()
            if not chat_session:
                raise HTTPException(status_code=404, detail="Session not found")

            chat_session.title = request.title
            session.commit()
            return {"status": "success", "title": request.title}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(json.dumps({"message": f"Error renaming session {session_id}: {e}", "status_code": 500}))
        raise HTTPException(status_code=500, detail="Internal server error")


async def delete_chat_session(session_id: str, current_user: User):
    """Deletes a specific chat session and all its messages."""
    try:
        with SessionLocal() as session:
            chat_session = session.query(ChatSession).filter_by(
                session_id=session_id, user_uuid=current_user.user_uuid
            ).first()
            if not chat_session:
                return {"status": "success", "message": "Session not found but considered deleted"}
            session.query(ChatHistory).filter_by(session_id=session_id).delete()
            session.delete(chat_session)
            session.commit()
            return {"status": "success", "message": "Session deleted"}
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to delete session.")


async def create_chat_session(current_user: User):
    """Creates a new chat session."""
    try:
        session_id = str(uuid.uuid4())
        with SessionLocal() as session:
            new_chat_session = ChatSession(
                session_id=session_id,
                user_uuid=current_user.user_uuid,
                title="New Conversation",
                context_data=None
            )
            session.add(new_chat_session)
            session.commit()
            return {"session_id": session_id}
    except Exception as e:
        logger.error(json.dumps({"message": f"Failed to create chat session: {e}", "status_code": 500}))
        raise HTTPException(status_code=500, detail="Failed to create new chat session.")


async def run_graph_task(task_id: str, state_input: dict):
    """Executes the chat graph in the background and sets Redis output."""
    try:
        full_response = []
        config = {
            "run_name": "budai-agent-chat",
            "configurable": {"thread_id": state_input.get("session_id", str(uuid.uuid4()))}
        }
        langfuse_handler = CallbackHandler()
        config["metadata"] = {
            "session_id": state_input.get("session_id", "default"),
            "user_id": str(state_input.get("user_uuid", "system"))
        }
        config["callbacks"] = [langfuse_handler]

        async for event in budai_app.astream_events(state_input, version="v2", config=config):
            if event.get("event") == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    delta = chunk.content
                    if isinstance(delta, str):
                        full_response.append(delta)
                    elif isinstance(delta, list):
                        for part in delta:
                            if isinstance(part, str):
                                full_response.append(part)
                            elif isinstance(part, dict) and "text" in part:
                                full_response.append(part["text"])

        redis_client.set(f"task:{task_id}", json.dumps({
            "status": "completed",
            "result": "".join(full_response)
        }), ex=3600)
    except Exception as e:
        logger.error(json.dumps({"message": f"Task {task_id} failed: {e}", "status_code": 500}))
        redis_client.set(f"task:{task_id}", json.dumps({
            "status": "failed", "error": str(e)
        }), ex=3600)
    finally:
        GlobalLLMManager.release()


async def async_chat(request: ChatRequest, background_tasks: BackgroundTasks, current_user: User):
    """Offloads chat execution to background task."""
    GlobalLLMManager.try_acquire()

    chat_history = await asyncio.to_thread(get_session_history, current_user.user_uuid, request.session_id)
    initial_state = {
        "messages": chat_history + [HumanMessage(content=request.input)],
        "user_uuid": str(current_user.user_uuid),
        "session_id": request.session_id,
        "active_account_id": request.active_account_id or None,
        "is_explanation": False
    }
    task_id = str(uuid.uuid4())
    redis_client.set(f"task:{task_id}", json.dumps({"status": "pending"}), ex=3600)
    background_tasks.add_task(run_graph_task, task_id, initial_state)

    return {"task_id": task_id, "status": "queued"}


async def stream_chat(request: StreamChatRequest, current_user: User):
    """Streams a chat response back using Server-Sent Events."""
    GlobalLLMManager.try_acquire()

    thread_id = request.session_id
    if not thread_id:
        from datetime import timedelta
        from models.database_models import utc_now
        try:
            with SessionLocal() as s:
                recent_sess = s.query(ChatSession).filter(
                    ChatSession.user_uuid == str(current_user.user_uuid),
                    ChatSession.last_updated >= utc_now() - timedelta(minutes=2)
                ).order_by(ChatSession.last_updated.desc()).first()
                if recent_sess:
                    thread_id = recent_sess.session_id
        except Exception as e:
            logger.warning(f"Failed to lookup recent session: {e}")
        if not thread_id:
            thread_id = str(uuid.uuid4())

    resume_command = None
    user_input = ""
    if request.messages:
        last_msg = request.messages[-1]
        if request.htil_response:
            payload = request.htil_response
            user_message = payload.get("user_message", "[Decision provided]")
            resume_command = Command(resume={
                "decisions": [{"type": "respond", "message": user_message}]
            })
            user_input = f"[RESUMING]: {user_message}"
        else:
            if last_msg.content:
                user_input = last_msg.content
            elif last_msg.parts:
                user_input = "".join([p.get("text", "") for p in last_msg.parts if p.get("type") == "text"])

    logger.info(f"Stream request: '{user_input[:50]}...', Session: {thread_id}")

    chat_history = await asyncio.to_thread(get_session_history, current_user.user_uuid, thread_id)

    effective_user_input = user_input
    if request.verbosity == "concise":
        effective_user_input = f"{user_input}\n[STYLE: Concise mode. Deliver precise data and actionable takeaways. Keep brief.]"
    elif request.verbosity == "deep_dive":
        effective_user_input = f"{user_input}\n[STYLE: Deep Dive mode. Provide comprehensive quantitative breakdown, multi-step agent audits, and mathematical rationale.]"

    initial_state = {
        "messages": chat_history + [HumanMessage(content=effective_user_input)],
        "user_uuid": str(current_user.user_uuid),
        "session_id": thread_id,
        "active_account_id": request.active_account_id or None,
        "is_explanation": False
    }

    if not resume_command:
        state_to_exec = {
            "user_uuid": current_user.user_uuid,
            "session_id": thread_id,
            "user_input": effective_user_input
        }
        await execute_chat_graph_async(state_to_exec)
        thread_id = state_to_exec.get("session_id") or thread_id
        initial_state["session_id"] = thread_id

    async def generate_response():
        config = {"run_name": "budai-agent-chat", "configurable": {"thread_id": thread_id}}
        langfuse_handler = CallbackHandler()
        config["metadata"] = {"session_id": request.session_id or "default", "user_id": str(current_user.user_uuid)}
        config["callbacks"] = [langfuse_handler]

        has_checkpoint_history = False
        try:
            current_state = budai_app.get_state(config)
            if current_state and current_state.values and "messages" in current_state.values:
                state_msgs = current_state.values["messages"]
                if state_msgs:
                    has_checkpoint_history = True
                tool_call_ids_responded = {getattr(m, "tool_call_id", None) for m in state_msgs if isinstance(m, ToolMessage)}
                missing_tool_responses = []
                for m in state_msgs:
                    if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                        for tc in m.tool_calls:
                            tc_id = tc.get("id")
                            if tc_id and tc_id not in tool_call_ids_responded:
                                missing_tool_responses.append(
                                    ToolMessage(
                                        content="[Tool execution encountered an internal error or was interrupted. Proceeding with available context.]",
                                        tool_call_id=tc_id,
                                        name=tc.get("name", "unknown_tool")
                                    )
                                )
                if missing_tool_responses:
                    logger.warning(f"Repairing {len(missing_tool_responses)} orphaned tool calls in session {thread_id}")
                    budai_app.update_state(config, {"messages": missing_tool_responses})
        except Exception as e:
            logger.warning(f"Checkpointer state repair check skipped: {e}")

        if resume_command:
            input_data = resume_command
        elif has_checkpoint_history:
            input_data = {
                "messages": [HumanMessage(content=effective_user_input)],
                "user_uuid": str(current_user.user_uuid),
                "session_id": thread_id,
                "active_account_id": request.active_account_id or None,
                "is_explanation": False
            }
        else:
            input_data = initial_state

        queue = asyncio.Queue()
        full_assistant_response = []
        full_reasoning_response = []
        tokens_generated = {"count": 0}
        time_metrics = {"start": time.time(), "ttft": None}
        msg_id = request.messageId or f"msg_{uuid.uuid4().hex[:8]}"
        reasoning_msg_id = f"reasoning_{uuid.uuid4().hex[:8]}"

        token_stats = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0
        }

        async def process_events():
            try:
                async for event in budai_app.astream_events(input_data, version="v2", config=config):
                    kind = event.get("event")
                    if kind == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "additional_kwargs"):
                            reasoning = chunk.additional_kwargs.get("reasoning_content")
                            if reasoning:
                                if time_metrics["ttft"] is None:
                                    time_metrics["ttft"] = int((time.time() - time_metrics["start"]) * 1000)
                                tokens_generated["count"] += 1
                                token_stats["reasoning_tokens"] += 1
                                full_reasoning_response.append(reasoning)
                                sys.stderr.write(f"\033[90m{reasoning}\033[0m")
                                sys.stderr.flush()
                                await queue.put(f'r:{json.dumps(reasoning)}\n')

                        if chunk and hasattr(chunk, "content") and chunk.content:
                            delta = chunk.content
                            tokens_generated["count"] += 1
                            token_stats["output_tokens"] += 1

                            if delta:
                                if time_metrics["ttft"] is None:
                                    time_metrics["ttft"] = int((time.time() - time_metrics["start"]) * 1000)
                                full_assistant_response.append(delta)
                                sys.stderr.write(f"\033[96m{delta}\033[0m")
                                sys.stderr.flush()
                                await queue.put(f'0:{json.dumps(delta)}\n')

                    elif kind == "on_chat_model_end":
                        output = event.get("data", {}).get("output")
                        if output:
                            usage = getattr(output, "usage_metadata", None)
                            if usage and isinstance(usage, dict):
                                if usage.get("input_tokens"):
                                    token_stats["input_tokens"] += usage.get("input_tokens", 0)
                                if usage.get("output_tokens") and token_stats["output_tokens"] == 0:
                                    token_stats["output_tokens"] = usage.get("output_tokens", 0)
                            elif hasattr(output, "response_metadata"):
                                resp_meta = output.response_metadata or {}
                                token_usage = resp_meta.get("token_usage", {})
                                if token_usage.get("prompt_tokens"):
                                    token_stats["input_tokens"] += token_usage.get("prompt_tokens", 0)

                    elif kind == "on_tool_start":
                        tool_name = event.get("name")
                        agent_map = {
                            "call_analyser_agent": ("AnalyserAgent", "Analyzing historical spending & cashflow"),
                            "call_forecaster_agent": ("ForecasterWorker", "Simulating stochastic balance trajectories"),
                            "call_categorizer_agent": ("CategorizerAgent", "Enriching taxonomy & ONS macro tags"),
                            "call_health_agent": ("HealthWorker", "Evaluating liquidity reserves & safety runway"),
                            "call_memory_service": ("MemoryService", "Retrieving qualitative profile memory"),
                            "call_market_agent": ("MarketWorker", "Fetching macroeconomic & market telemetry"),
                            "call_scenario_agent": ("ScenarioWorker", "Computing structural financial scenarios"),
                            "call_bucket_transfer_agent": ("BucketAgent", "Calculating virtual capital allocations"),
                            "call_anomaly_detection_agent": ("AnomalyWorker", "Screening transactions for surge variance"),
                            "call_debt_management_agent": ("DebtWorker", "Modeling liability payoff trajectories"),
                            "call_goal_tracking_agent": ("GoalWorker", "Auditing savings goal milestone progress"),
                            "call_income_detection_agent": ("IncomeWorker", "Profiling recurring payroll rhythms"),
                            "call_lifestyle_clustering_agent": ("LifestyleWorker", "Clustering behavioral spending archetypes"),
                            "call_monte_carlo_simulation_agent": ("MonteCarloEngine", "Running Bates jump-diffusion simulations"),
                            "call_subscription_detection_agent": ("SubscriptionWorker", "Auditing active recurring billing"),
                            "get_connected_accounts_orchestrator": ("AccountWorker", "Querying verified bank accounts"),
                            "get_liability_horizon_orchestrator": ("LiabilityWorker", "Calculating liability horizon & scheduled sweeps"),
                            "query_transactions": ("AnalyserAgent", "Querying verified transaction records"),
                            "aggregate_financial_data": ("AnalyserAgent", "Analyzing historical spending & cashflow"),
                            "get_user_lifestyle_profile_wrapper": ("LifestyleWorker", "Profiling behavioral spending personas"),
                            "get_semantic_anomalies_wrapper": ("AnomalyWorker", "Screening semantic transaction anomalies"),
                            "get_lifestyle_trajectory_wrapper": ("LifestyleWorker", "Analyzing 3-month lifestyle trajectory"),
                            "benchmark_persona_budget_wrapper": ("LifestyleWorker", "Benchmarking spending against persona averages"),
                            "predict_impulse_vulnerability_wrapper": ("LifestyleWorker", "Predicting impulse spending windows"),
                            "get_upcoming_subscriptions_wrapper": ("SubscriptionWorker", "Auditing upcoming recurring bills"),
                        }
                        if tool_name in agent_map:
                            agent_name, agent_action = agent_map[tool_name]
                            await queue.put(f'8:[{json.dumps({"type": "subagent_step", "agent": agent_name, "action": agent_action, "status": "running"})}]\n')
                            await queue.put(f'8:[{json.dumps({"type": "thinking_context", "status": f"Consulting {agent_name}..."})}]\n')
                        elif tool_name == "generate_ui_chart":
                            tool_input = event.get("data", {}).get("input", {})
                            tool_call = {
                                "toolCallId": f"call_{str(uuid.uuid4())[:8]}",
                                "toolName": "generate_ui_chart",
                                "args": tool_input
                            }
                            await queue.put(f'9:[{json.dumps(tool_call)}]\n')
                            await queue.put(f'8:[{json.dumps({"type": "thinking_context", "status": "Drawing Chart"})}]\n')
                        elif tool_name == "ask_user":
                            tool_input = event.get("data", {}).get("input", {})
                            question = tool_input.get("question")
                            if question:
                                tool_call = {
                                    "toolCallId": f"call_{str(uuid.uuid4())[:8]}",
                                    "toolName": "ask_user",
                                    "args": {"question": question}
                                }
                                await queue.put(f'9:[{json.dumps(tool_call)}]\n')

                    elif kind == "on_tool_end":
                        tool_name = event.get("name")
                        agent_end_map = {
                            "call_analyser_agent": "AnalyserAgent",
                            "call_forecaster_agent": "ForecasterWorker",
                            "call_categorizer_agent": "CategorizerAgent",
                            "call_health_agent": "HealthWorker",
                            "call_memory_service": "MemoryService",
                            "call_market_agent": "MarketWorker",
                            "call_scenario_agent": "ScenarioWorker",
                            "call_bucket_transfer_agent": "BucketAgent",
                            "call_anomaly_detection_agent": "AnomalyWorker",
                            "call_debt_management_agent": "DebtWorker",
                            "call_goal_tracking_agent": "GoalWorker",
                            "call_income_detection_agent": "IncomeWorker",
                            "call_lifestyle_clustering_agent": "LifestyleWorker",
                            "call_monte_carlo_simulation_agent": "MonteCarloEngine",
                            "call_subscription_detection_agent": "SubscriptionWorker",
                            "get_connected_accounts_orchestrator": "AccountWorker",
                            "get_liability_horizon_orchestrator": "LiabilityWorker",
                            "query_transactions": "AnalyserAgent",
                            "aggregate_financial_data": "AnalyserAgent",
                            "get_user_lifestyle_profile_wrapper": "LifestyleWorker",
                            "get_semantic_anomalies_wrapper": "AnomalyWorker",
                            "get_lifestyle_trajectory_wrapper": "LifestyleWorker",
                            "benchmark_persona_budget_wrapper": "LifestyleWorker",
                            "predict_impulse_vulnerability_wrapper": "LifestyleWorker",
                            "get_upcoming_subscriptions_wrapper": "SubscriptionWorker",
                        }
                        if tool_name in agent_end_map:
                            agent_name = agent_end_map[tool_name]
                            await queue.put(f'8:[{json.dumps({"type": "subagent_step", "agent": agent_name, "status": "completed"})}]\n')

            except Exception as e:
                logger.error(json.dumps({"message": f"Event processing failed: {e}", "status_code": 500}))
                await queue.put(e)

        async def keep_alive():
            try:
                while True:
                    await asyncio.sleep(4.0)
                    await queue.put(": keep-alive\n\n")
            except asyncio.CancelledError:
                pass

        tasks = [
            asyncio.create_task(process_events()),
            asyncio.create_task(keep_alive())
        ]

        async def wait_all():
            await tasks[0]
            if len(tasks) > 1 and not tasks[1].done():
                tasks[1].cancel()
            await asyncio.sleep(0.1)

            compute_time_ms = int((time.time() - time_metrics["start"]) * 1000)
            ttft_ms = time_metrics["ttft"] if time_metrics["ttft"] is not None else compute_time_ms
            from services.llm_factory import get_model_context_limit

            prompt_tokens = token_stats["input_tokens"] if token_stats["input_tokens"] > 0 else max(80, len(effective_user_input) // 3)
            completion_tokens = token_stats["output_tokens"] + token_stats["reasoning_tokens"]
            if completion_tokens == 0:
                completion_tokens = tokens_generated["count"]
            total_tokens = prompt_tokens + completion_tokens

            telemetry_data = {
                "type": "telemetry",
                "ttft_ms": ttft_ms,
                "compute_time_ms": compute_time_ms,
                "tokens": total_tokens,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "reasoning_tokens": token_stats["reasoning_tokens"],
                "context_limit": get_model_context_limit(),
                "verbosity": request.verbosity or "balanced"
            }
            await queue.put(f'8:[{json.dumps(telemetry_data)}]\n')

            usage_data = {
                "finishReason": "stop",
                "usage": {
                    "promptTokens": prompt_tokens,
                    "completionTokens": completion_tokens,
                    "totalTokens": total_tokens
                }
            }
            await queue.put(f'd:{json.dumps(usage_data)}\n')
            await queue.put(None)

        try:
            asyncio.create_task(wait_all())

            yield f'data: {json.dumps({"type": "start", "messageId": msg_id})}\n\n'
            yield f'data: {json.dumps({"type": "text-start", "id": msg_id})}\n\n'
            has_reasoning = False
            finish_item = None

            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=4.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue

                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item

                if isinstance(item, str):
                    if item.startswith(":"):
                        yield item
                        continue
                    try:
                        if item.startswith("0:"):
                            content = json.loads(item[2:].strip())
                            item = f'data: {json.dumps({"type": "text-delta", "id": msg_id, "delta": content})}\n\n'
                            yield item
                        elif item.startswith("r:"):
                            if not has_reasoning:
                                has_reasoning = True
                                yield f'data: {json.dumps({"type": "reasoning-start", "id": reasoning_msg_id})}\n\n'
                            content = json.loads(item[2:].strip())
                            item = f'data: {json.dumps({"type": "reasoning-delta", "id": reasoning_msg_id, "delta": content})}\n\n'
                            yield item
                        elif item.startswith("8:"):
                            content = json.loads(item[2:].strip())
                            item = f'data: {json.dumps({"type": "data-message_annotations", "data": content})}\n\n'
                            yield item
                        elif item.startswith("9:"):
                            content = json.loads(item[2:].strip())
                            item = f'data: {json.dumps({"type": "data-tool_calls", "data": content})}\n\n'
                            yield item
                        elif item.startswith("d:"):
                            usage_data = json.loads(item[2:].strip())
                            u = usage_data.get("usage", {})
                            finish_item = f'data: {json.dumps({"type": "finish", "finishReason": usage_data.get("finishReason", "stop"), "messageMetadata": {"usage": {"inputTokens": u.get("promptTokens", 0), "outputTokens": u.get("completionTokens", 0), "totalTokens": u.get("totalTokens", 0)}}})}\n\n'
                    except Exception as parse_e:
                        logger.error(f"Failed to convert stream format: {parse_e}. Item was: {item!r}")

            if has_reasoning:
                yield f'data: {json.dumps({"type": "reasoning-end", "id": reasoning_msg_id})}\n\n'
            yield f'data: {json.dumps({"type": "text-end", "id": msg_id})}\n\n'
            yield f'data: {json.dumps({"type": "finish-step"})}\n\n'
            if finish_item:
                yield finish_item
            yield 'data: [DONE]\n\n'

            for t in tasks:
                if not t.done():
                    t.cancel()

            final_response = "".join(full_assistant_response)
            if final_response.strip():
                try:
                    def _save_history():
                        with SessionLocal() as session:
                            compute_time = int((time.time() - time_metrics["start"]) * 1000)
                            ttft = time_metrics["ttft"] if time_metrics["ttft"] is not None else compute_time
                            prompt_toks = token_stats["input_tokens"] if token_stats["input_tokens"] > 0 else max(80, len(effective_user_input) // 3)
                            comp_toks = token_stats["output_tokens"] + token_stats["reasoning_tokens"]
                            if comp_toks == 0:
                                comp_toks = tokens_generated["count"]
                            total_toks = prompt_toks + comp_toks
                            new_msg = ChatHistory(
                                user_uuid=current_user.user_uuid,
                                session_id=thread_id,
                                role="assistant",
                                content=final_response,
                                ttft_ms=ttft,
                                compute_time_ms=compute_time,
                                tokens=total_toks,
                                reasoning_content="".join(full_reasoning_response) if full_reasoning_response else None
                            )
                            session.add(new_msg)
                            session.commit()
                    await asyncio.to_thread(_save_history)
                except Exception as e:
                    logger.error(json.dumps({"message": f"Failed to save assistant msg: {e}", "status_code": 500}))

        except asyncio.CancelledError:
            logger.warning("Stream cancelled by client disconnect. Cancelling graph execution.")
            for t in tasks:
                if not t.done():
                    t.cancel()
        except Exception as e:
            logger.error(json.dumps({"message": f"Stream error: {e}", "status_code": 500}))
        finally:
            GlobalLLMManager.release()

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "x-vercel-ai-ui-message-stream": "v1",
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


async def chat(request: ChatRequest, current_user: User):
    """Initiates a streaming chat request (Legacy v2 streaming pattern)."""
    GlobalLLMManager.try_acquire()

    chat_history = await asyncio.to_thread(get_session_history, current_user.user_uuid, request.session_id)
    initial_state = {
        "messages": chat_history + [HumanMessage(content=request.input)],
        "user_uuid": str(current_user.user_uuid),
        "session_id": request.session_id,
        "active_account_id": request.active_account_id or None,
        "is_explanation": False
    }
    await execute_chat_graph_async({
        "user_uuid": current_user.user_uuid,
        "session_id": request.session_id,
        "user_input": request.input
    })

    async def generate():
        config = {
            "run_name": "budai-agent-chat",
            "configurable": {"thread_id": request.session_id or str(uuid.uuid4())}
        }
        langfuse_handler = CallbackHandler()
        config["metadata"] = {"session_id": request.session_id or "default", "user_id": str(current_user.user_uuid)}
        config["callbacks"] = [langfuse_handler]
        queue = asyncio.Queue()

        async def consume_messages(stream_handle, is_subgraph=False, name="Supervisor"):
            try:
                async for message in stream_handle.messages:
                    node_label = f"[{name}::{message.node}]" if is_subgraph else f"[{message.node}]"
                    async for delta in message.reasoning:
                        await queue.put(f'0:{json.dumps(f"[thinking] {node_label} {delta}")}\n')
                        await asyncio.sleep(0.01)
                    async for delta in message.text:
                        if not delta:
                            continue
                        await queue.put(f'0:{json.dumps(delta)}\n')
            except Exception as e:
                if not is_subgraph:
                    await queue.put(e)

        try:
            async with await budai_app.astream_events(initial_state, version="v3", config=config) as stream:
                tasks = [
                    asyncio.create_task(consume_messages(stream))
                ]
                async def wait_all():
                    await stream.output()
                    await asyncio.sleep(0.5)
                    await queue.put(None)
                tasks.append(asyncio.create_task(wait_all()))
                yield f'data: {json.dumps({"type": "text-start", "id": "msg_0"})}\n\n'

                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    if isinstance(item, str) and item.startswith("0:"):
                        content = json.loads(item[2:].strip())
                        yield f'data: {json.dumps({"type": "text-delta", "id": "msg_0", "delta": content})}\n\n'

                yield f'data: {json.dumps({"type": "text-end", "id": "msg_0"})}\n\n'
                yield f'data: {json.dumps({"type": "finish-step"})}\n\n'
                yield 'data: [DONE]\n\n'
        except Exception as e:
            logger.error(json.dumps({"message": f"Generate error: {e}", "status_code": 500}))
        finally:
            GlobalLLMManager.release()

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Accel-Buffering": "no",
            "x-vercel-ai-ui-message-stream": "v1",
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
