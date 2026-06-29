import sys
from models.database_models import ChatHistory
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List
from middleware.auth_middleware import get_current_user
from models.database_models import User, ChatSession
from schemas.api_schema import ChatRequest, ExplanationRequest, ChatSessionResponse, ChatMessageResponse, StreamChatRequest, ChatSessionRenameRequest
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

chat_router = APIRouter(prefix="/api/chat", tags=["chat"])


async def run_graph_task(task_id: str, state_input: dict):
    """Executes the graph asynchronously and stores the result using v3 protocol in Redis."""
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


@chat_router.post("/async")
async def async_chat(request: ChatRequest, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    """Initiates an asynchronous chat request."""
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


@chat_router.post("/stream")
async def stream_chat(request: StreamChatRequest, current_user: User = Depends(get_current_user)):
    """Streams a chat response back to the client using LangChain v3 Event Streaming projections."""
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
                            logger.info(item.strip())
                            yield item
                        elif item.startswith("9:"):
                            content = json.loads(item[2:].strip())
                            item = f'data: {json.dumps({"type": "data-tool_calls", "data": content})}\n\n'
                            logger.info(item.strip())
                            yield item
                        elif item.startswith("d:"):
                            usage_data = json.loads(item[2:].strip())
                            finish_item = f'data: {json.dumps({"type": "finish", "finishReason": usage_data.get("finishReason", "stop")})}\n\n'
                            logger.info(
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
            raise
        except Exception as e:
            logger.error(f"Stream error: {e}")

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


@chat_router.get("/status/{task_id}")
def get_task_status(task_id: str):
    """Retrieves the status of an asynchronous chat task from Redis."""
    task_info_raw = redis_client.get(f"task:{task_id}")
    if not task_info_raw:
        raise HTTPException(status_code=404, detail="Task not found")
    return json.loads(task_info_raw)


@chat_router.post("/sessions")
def create_chat_session(current_user: User = Depends(get_current_user)):
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
        raise HTTPException(
            status_code=500, detail="Failed to create new chat session.")


@chat_router.get("/sessions", response_model=List[ChatSessionResponse])
def list_chat_sessions(current_user: User = Depends(get_current_user)):
    """Lists all chat sessions for the current user."""
    try:
        with SessionLocal() as session:
            sessions = session.query(ChatSession).filter_by(
                user_uuid=current_user.user_uuid).order_by(ChatSession.last_updated.desc()).all()
            return [ChatSessionResponse(session_id=s.session_id, title=s.title or "New Conversation", last_updated=s.last_updated, context_data=s.context_data) for s in sessions]
    except Exception:
        raise HTTPException(
            status_code=500, detail="Failed to retrieve chat history.")


@chat_router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
def get_chat_session(session_id: str, current_user: User = Depends(get_current_user)):
    """Retrieves a specific chat session and its messages."""
    try:
        with SessionLocal() as session:
            chat_session = session.query(ChatSession).filter_by(
                session_id=session_id, user_uuid=current_user.user_uuid).first()
            if not chat_session:
                raise HTTPException(
                    status_code=404, detail="Session not found.")
            messages = session.query(ChatHistory).filter_by(
                session_id=session_id).order_by(ChatHistory.timestamp.asc()).all()

            message_responses = [ChatMessageResponse(
                role=m.role, content=m.content, reasoning_content=m.reasoning_content, timestamp=m.timestamp) for m in messages]

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
        raise HTTPException(
            status_code=500, detail="Failed to retrieve session details.")


@chat_router.post("")
async def chat(request: ChatRequest, current_user: User = Depends(get_current_user)):
    """Initiates a standard streaming chat request."""
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
                        # check if it's a tool message or AIMessage with tool calls
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
                asyncio.create_task(wait_all())
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
                                logger.info(item.strip())
                                yield item
                            elif item.startswith("9:"):
                                content = json.loads(item[2:].strip())
                                item = f'data: {json.dumps({"type": "data-tool_calls", "data": content})}\n\n'
                                logger.info(item.strip())
                                yield item
                            elif item.startswith("d:"):
                                usage_data = json.loads(item[2:].strip())
                                finish_item = f'data: {json.dumps({"type": "finish", "finishReason": usage_data.get("finishReason", "stop")})}\n\n'
                                logger.info(
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

@chat_router.patch("/sessions/{session_id}")
def rename_chat_session(session_id: str, request: ChatSessionRenameRequest, current_user: User = Depends(get_current_user)):
    """Renames a specific chat session."""
    try:
        with SessionLocal() as session:
            chat_session = session.query(ChatSession).filter_by(
                session_id=session_id, user_uuid=current_user.user_uuid).first()
            if not chat_session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            chat_session.title = request.title
            session.commit()
            return {"status": "success", "title": request.title}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error renaming session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@chat_router.delete("/sessions/{session_id}")
def delete_chat_session(session_id: str, current_user: User = Depends(get_current_user)):
    """Deletes a specific chat session."""
    try:
        with SessionLocal() as session:
            chat_session = session.query(ChatSession).filter_by(
                session_id=session_id, user_uuid=current_user.user_uuid).first()
            if not chat_session:
                return {"status": "success", "message": "Session not found but considered deleted"}
            session.query(ChatHistory).filter_by(
                session_id=session_id).delete()
            session.delete(chat_session)
            session.commit()
            return {"status": "success", "message": "Session deleted"}
    except Exception:
        raise HTTPException(
            status_code=500, detail="Failed to delete session.")
