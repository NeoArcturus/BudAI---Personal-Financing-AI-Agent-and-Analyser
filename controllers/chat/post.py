import sys
from models.database_models import ChatHistory
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List
from middleware.auth_middleware import get_current_user
from models.database_models import User, ChatSession
from schemas.api_schema import ChatRequest, ChatSessionResponse, ChatMessageResponse, StreamChatRequest, ChatSessionRenameRequest
from services.orchestrator_graph import execute_chat_graph_async, get_session_history, budai_app
from config import SessionLocal
import asyncio
import uuid
import json
from services.logger_setup import get_core_logger
from langgraph.types import Command
from langchain_core.messages import HumanMessage

logger = get_core_logger(__name__)

from config import SessionLocal, redis_client

async def run_graph_task(task_id: str, state_input: dict):
    """
    Executes the LangGraph chat model asynchronously and stores the full generated
    result using the v3 protocol in Redis for later retrieval by the client.
    
    Args:
        task_id (str): The UUID of the background task tracking this execution.
        state_input (dict): The initial state containing message history and user context.
    """
    try:
        full_response = []
        config = {"configurable": {"thread_id": state_input.get(
            "session_id", str(uuid.uuid4()))}}

        async with await budai_app.astream_events(state_input, version="v3", config=config) as stream:
            async for message in stream.messages:
                async for delta in message.text:
                    full_response.append(delta)

        redis_client.set(f"task:{task_id}", json.dumps({
            "status": "completed",
            "result": "".join(full_response)
        }), ex=3600)
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        redis_client.set(f"task:{task_id}", json.dumps({
            "status": "failed", "error": str(e)
        }), ex=3600)
    finally:
        from services.llm_manager import GlobalLLMManager
        GlobalLLMManager.release()
async def async_chat(request: ChatRequest, background_tasks: BackgroundTasks, current_user: User):
    """
    Initiates an asynchronous chat request. This offloads the LangGraph execution
    to a background task and returns a task_id immediately for polling.
    
    Args:
        request (ChatRequest): The chat input payload.
        background_tasks (BackgroundTasks): FastAPI background tasks dependency.
        current_user (User): The authenticated user making the request.
        
    Returns:
        dict: The queued task ID and status.
    """
    from services.llm_manager import GlobalLLMManager
    GlobalLLMManager.try_acquire()
    
    chat_history = await asyncio.to_thread(get_session_history, current_user.user_uuid, request.session_id)
    initial_state = {
        "messages": chat_history + [HumanMessage(content=request.input)],
        "user_uuid": str(current_user.user_uuid),
        "session_id": request.session_id,
        "active_account_id": request.active_account_id or "ALL",
        "is_explanation": False
    }
    task_id = str(uuid.uuid4())
    redis_client.set(f"task:{task_id}", json.dumps({"status": "pending"}), ex=3600)
    background_tasks.add_task(run_graph_task, task_id, initial_state)
        
    return {"task_id": task_id, "status": "queued"}

async def stream_chat(request: StreamChatRequest, current_user: User):
    """
    Streams a chat response back to the client using LangChain v3 Event Streaming projections.
    Handles intermediate steps like reasoning output, tool calls (e.g. chart rendering),
    and saves the final message to the database once the stream completes.
    
    Args:
        request (StreamChatRequest): The chat input containing the active messages list.
        current_user (User): The authenticated user making the request.
        
    Returns:
        StreamingResponse: A Server-Sent Events (SSE) stream of JSON chunks.
    """
    from services.llm_manager import GlobalLLMManager
    GlobalLLMManager.try_acquire()
    
    thread_id = request.session_id or str(uuid.uuid4())

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
                user_input = "".join([p.get("text", "")
                                     for p in last_msg.parts if p.get("type") == "text"])

    logger.info(
        f"Stream request: '{user_input[:50]}...', Session: {thread_id}")

    chat_history = await asyncio.to_thread(get_session_history, current_user.user_uuid, thread_id)
    initial_state = {
        "messages": chat_history + [HumanMessage(content=user_input)],
        "user_uuid": str(current_user.user_uuid),
        "session_id": thread_id,
        "active_account_id": request.active_account_id or "ALL",
        "is_explanation": False
    }

    if not resume_command:

        await execute_chat_graph_async({
            "user_uuid": current_user.user_uuid,
            "session_id": thread_id,
            "user_input": user_input
        })

    async def generate_response():
        config = {"configurable": {"thread_id": thread_id}}
        input_data = resume_command if resume_command else initial_state
        queue = asyncio.Queue()
        sent_cache_ids = set()
        full_assistant_response = []
        full_reasoning_response = []
        tokens_generated = {"count": 0}
        import time
        time_metrics = {"start": time.time(), "ttft": None}
        msg_id = request.messageId or f"msg_{uuid.uuid4().hex[:8]}"
        reasoning_msg_id = f"reasoning_{uuid.uuid4().hex[:8]}"

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
                                full_reasoning_response.append(reasoning)
                                sys.stderr.write(f"\033[90m{reasoning}\033[0m")
                                sys.stderr.flush()
                                await queue.put(f'r:{json.dumps(reasoning)}\n')
                    
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            delta = chunk.content
                            tokens_generated["count"] += 1
                        
                            if delta:
                                if time_metrics["ttft"] is None:
                                    time_metrics["ttft"] = int((time.time() - time_metrics["start"]) * 1000)
                                full_assistant_response.append(delta)
                                sys.stderr.write(f"\033[96m{delta}\033[0m")
                                sys.stderr.flush()
                                await queue.put(f'0:{json.dumps(delta)}\n')
                
                    elif kind == "on_tool_start":
                        tool_name = event.get("name")
                        if tool_name == "render_ui_chart":
                            tool_input = event.get("data", {}).get("input", {})
                            chart_type = tool_input.get("chart_type")
                            cache_id = tool_input.get("cache_id")
                            if cache_id and chart_type and cache_id not in sent_cache_ids:
                                tool_call = {
                                    "toolCallId": f"call_{str(uuid.uuid4())[:8]}",
                                    "toolName": "render_ui_chart",
                                    "args": {"chart_type": chart_type, "cache_id": cache_id}
                                }
                                await queue.put(f'9:[{json.dumps(tool_call)}]\n')
                                await queue.put(f'8:[{json.dumps({"type": "global_refresh_signal", "chart_type": chart_type})}]\n')
                                await queue.put(f'8:[{json.dumps({"type": "thinking_context", "status": "Drawing"})}]\n')
                                sent_cache_ids.add(cache_id)
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

            except Exception as e:
                logger.error(f"Event processing failed: {e}")
                await queue.put(e)

        async def keep_alive():
            try:
                while True:
                    await queue.put(f'8:[{json.dumps({"type": "thinking_context", "status": "Thinking"})}]\n')
                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                pass

        tasks = [
            asyncio.create_task(process_events()),
            asyncio.create_task(keep_alive())
        ]

        async def wait_all():
            await tasks[0]
            await asyncio.sleep(0.5)
        
            compute_time_ms = int((time.time() - time_metrics["start"]) * 1000)
            ttft_ms = time_metrics["ttft"] if time_metrics["ttft"] is not None else compute_time_ms
            telemetry_data = {
                "type": "telemetry",
                "ttft_ms": ttft_ms,
                "compute_time_ms": compute_time_ms,
                "tokens": tokens_generated["count"]
            }
            await queue.put(f'8:[{json.dumps(telemetry_data)}]\n')
        
            usage_data = {"finishReason": "stop", "usage": {
                "completionTokens": tokens_generated["count"], "promptTokens": 0}}
            await queue.put(f'd:{json.dumps(usage_data)}\n')
            await queue.put(None)

        try:
            asyncio.create_task(wait_all())
        
            yield f'data: {json.dumps({"type": "start", "messageId": msg_id})}\n\n'
            yield f'data: {json.dumps({"type": "text-start", "id": msg_id})}\n\n'
            has_reasoning = False
            finish_item = None

            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item

                if isinstance(item, str):
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
                            logger.debug(item.strip())
                            yield item
                        elif item.startswith("9:"):
                            content = json.loads(item[2:].strip())
                            item = f'data: {json.dumps({"type": "data-tool_calls", "data": content})}\n\n'
                            logger.debug(item.strip())
                            yield item
                        elif item.startswith("d:"):
                            usage_data = json.loads(item[2:].strip())
                            finish_item = f'data: {json.dumps({"type": "finish", "finishReason": usage_data.get("finishReason", "stop")})}\n\n'
                            logger.debug(
                                f"Captured finish: {finish_item.strip()}")
                    except Exception as parse_e:
                        logger.error(
                            f"Failed to convert stream format: {parse_e}. Item was: {item!r}")

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
                        from config import SessionLocal
                        from models.database_models import ChatHistory
                        with SessionLocal() as session:
                            compute_time = int((time.time() - time_metrics["start"]) * 1000)
                            ttft = time_metrics["ttft"] if time_metrics["ttft"] is not None else compute_time
                            new_msg = ChatHistory(user_uuid=current_user.user_uuid, session_id=thread_id,
                                                  role="assistant", content=final_response,
                                                  ttft_ms=ttft, compute_time_ms=compute_time, tokens=tokens_generated["count"],
                                                  reasoning_content="".join(full_reasoning_response) if full_reasoning_response else None)
                            session.add(new_msg)
                            session.commit()
                    await asyncio.to_thread(_save_history)
                except Exception as e:
                    logger.error(f"Failed to save assistant msg: {e}")

        except asyncio.CancelledError:
            logger.warning(
                "Stream cancelled by client disconnect. Cancelling graph execution.")
            for t in tasks:
                if not t.done():
                    t.cancel()
        except Exception as e:
            logger.error(f"Stream error: {e}")
        finally:
            from services.llm_manager import GlobalLLMManager
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


async def create_chat_session(current_user: User):
    """Creates a new chat session for the current user."""
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
        logger.error(f"Failed to create chat session: {e}")
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500, detail="Failed to create new chat session.")

async def chat(request: ChatRequest, current_user: User):
    """
    Initiates a standard streaming chat request (Legacy v2 streaming pattern).
    Uses a queue-based consumer architecture to read subgraph events and tool values.
    
    Args:
        request (ChatRequest): The chat input payload.
        current_user (User): The authenticated user making the request.
        
    Returns:
        StreamingResponse: An SSE stream containing text deltas and tool events.
    """
    from services.llm_manager import GlobalLLMManager
    GlobalLLMManager.try_acquire()
    
    chat_history = await asyncio.to_thread(get_session_history, current_user.user_uuid, request.session_id)
    initial_state = {
        "messages": chat_history + [HumanMessage(content=request.input)],
        "user_uuid": str(current_user.user_uuid),
        "session_id": request.session_id,
        "active_account_id": request.active_account_id or "ALL",
        "is_explanation": False
    }
    await execute_chat_graph_async({
        "user_uuid": current_user.user_uuid,
        "session_id": request.session_id,
        "user_input": request.input
    })

    async def generate():
        config = {"configurable": {
            "thread_id": request.session_id or str(uuid.uuid4())}}
        queue = asyncio.Queue()
        sent_cache_ids = set()

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
                        await queue.put(f'0:{json.dumps(delta)}\\n')
            except Exception as e:
                if not is_subgraph:
                    await queue.put(e)

        async def consume_subgraphs(stream_handle):
            try:
                async for subagent in stream_handle.subgraphs:
                    status_msg = {"type": "thinking_context",
                                  "status": f"ACTIVATING_{subagent.graph_name.upper()}"}
                    await queue.put(f'8:[{json.dumps(status_msg)}]\n')
                    asyncio.create_task(consume_messages(
                        subagent, is_subgraph=True, name=subagent.graph_name))
            except Exception:
                pass
    async def consume_values(stream_handle):
        try:
            async for snapshot in stream_handle.values:
                messages = snapshot.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if getattr(last_msg, "type", "") == "tool" or getattr(last_msg, "type", "") == "ai":
                        logger.info(
                            f"[State Update] Node returned message of type {getattr(last_msg, 'type', 'unknown')}: {getattr(last_msg, 'content', '')[:200]}...")

                cache_id = snapshot.get("cache_id")
                chart_type = snapshot.get("chart_type")
                if cache_id and chart_type and cache_id not in sent_cache_ids:
                    tool_call = {"toolCallId": f"call_{str(uuid.uuid4())[:8]}", "toolName": "render_ui_chart", "args": {
                        "chart_type": chart_type, "cache_id": cache_id}}
                    await queue.put(f'9:{json.dumps(tool_call)}\n')
                    sent_cache_ids.add(cache_id)
        except Exception:
            pass
        try:
            async with await budai_app.astream_events(initial_state, version="v3", config=config) as stream:
                tasks = [
                    asyncio.create_task(consume_messages(stream)),
                    asyncio.create_task(consume_values(stream)),
                    asyncio.create_task(consume_subgraphs(stream))
                ]
                async def wait_all():
                    await stream.output()
                    await asyncio.sleep(0.5)
                    await queue.put(None)
                tasks.append(asyncio.create_task(wait_all()))
                yield f'data: {json.dumps({"type": "text-start", "id": "msg_0"})}\n\n'

                finish_item = None
                has_reasoning = False

                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    if isinstance(item, str):
                            try:
                                if item.startswith("0:"):
                                    content = json.loads(item[2:].strip())
                                    item = f'data: {json.dumps({"type": "text-delta", "id": "msg_0", "delta": content})}\n\n'
                                    yield item
                                elif item.startswith("r:"):
                                    if not has_reasoning:
                                        has_reasoning = True
                                        yield f'data: {json.dumps({"type": "reasoning-start", "id": "reasoning_msg_0"})}\n\n'
                                    content = json.loads(item[2:].strip())
                                    item = f'data: {json.dumps({"type": "reasoning-delta", "id": "reasoning_msg_0", "delta": content})}\n\n'
                                    yield item
                                elif item.startswith("8:"):
                                    content = json.loads(item[2:].strip())
                                    item = f'data: {json.dumps({"type": "data-message_annotations", "data": content})}\n\n'
                                    logger.debug(item.strip())
                                    yield item
                                elif item.startswith("9:"):
                                    content = json.loads(item[2:].strip())
                                    item = f'data: {json.dumps({"type": "data-tool_calls", "data": content})}\n\n'
                                    logger.debug(item.strip())
                                    yield item
                                elif item.startswith("d:"):
                                    usage_data = json.loads(item[2:].strip())
                                    finish_item = f'data: {json.dumps({"type": "finish", "finishReason": usage_data.get("finishReason", "stop")})}\n\n'
                                    logger.debug(
                                        f"Captured finish: {finish_item.strip()}")
                            except Exception as parse_e:
                                logger.error(
                                    f"Failed to convert stream format: {parse_e}. Item was: {item!r}")

                if has_reasoning:
                    yield f'data: {json.dumps({"type": "reasoning-end", "id": "reasoning_msg_0"})}\n\n'
                yield f'data: {json.dumps({"type": "text-end", "id": "msg_0"})}\n\n'
                yield f'data: {json.dumps({"type": "finish-step"})}\n\n'
                if finish_item:
                    yield finish_item
                yield 'data: [DONE]\n\n'
        except Exception as e:
            logger.error(f"Generate error: {e}")
        finally:
            from services.llm_manager import GlobalLLMManager
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


