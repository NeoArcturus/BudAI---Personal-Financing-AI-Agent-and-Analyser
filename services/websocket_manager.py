import json
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
import redis.asyncio as aioredis
from config import REDIS_URL
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

class ConnectionManager:
    def __init__(self):
        # We don't need to keep sockets in memory for broadcasting if we use Redis PubSub
        # But we keep this for clean connection handling
        pass

    async def connect_and_listen(self, websocket: WebSocket, user_uuid: str):
        await websocket.accept()
        logger.info(json.dumps({"message": f"WebSocket Connected for user {user_uuid}", "status_code": 200}))
        
        redis_conn = await aioredis.from_url(REDIS_URL, decode_responses=True)
        pubsub = redis_conn.pubsub()
        channel_name = f"budai_events_{user_uuid}"
        
        await pubsub.subscribe(channel_name)
        
        try:
            # We need a task to listen to redis and send to websocket
            async def redis_listener():
                try:
                    async for message in pubsub.listen():
                        if message["type"] == "message":
                            await websocket.send_text(message["data"])
                except Exception as e:
                    logger.error(f"Redis listener error: {e}")
            
            listener_task = asyncio.create_task(redis_listener())
            
            # Keep websocket open and listen for client disconnects
            while True:
                data = await websocket.receive_text()
                # If we need to handle incoming messages from frontend, do it here
                
        except WebSocketDisconnect:
            logger.info(json.dumps({"message": f"WebSocket Disconnected for user {user_uuid}", "status_code": 200}))
        except Exception as e:
            logger.error(json.dumps({"message": f"WebSocket Error for {user_uuid}: {e}", "status_code": 500}))
        finally:
            listener_task.cancel()
            await pubsub.unsubscribe(channel_name)
            await redis_conn.aclose()

manager = ConnectionManager()
