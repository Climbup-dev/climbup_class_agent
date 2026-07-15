import redis
import json
from typing import Dict, List
from fastapi import WebSocket
from app.core.config import settings

# Initialize Redis client for pub/sub and state
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

class ClassroomManager:
    def __init__(self):
        # Local state mapping classroom_id -> active WebSockets
        self.active_connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, classroom_id: int, student_name: str):
        await websocket.accept()
        if classroom_id not in self.active_connections:
            self.active_connections[classroom_id] = []
        self.active_connections[classroom_id].append(websocket)
        
        # Add to Redis active list
        redis_client.sadd(f"classroom:{classroom_id}:students", student_name)

    def disconnect(self, websocket: WebSocket, classroom_id: int, student_name: str):
        if classroom_id in self.active_connections and websocket in self.active_connections[classroom_id]:
            self.active_connections[classroom_id].remove(websocket)
            if not self.active_connections[classroom_id]:
                del self.active_connections[classroom_id]
                
        # Remove from Redis active list
        redis_client.srem(f"classroom:{classroom_id}:students", student_name)

    async def broadcast_to_classroom(self, classroom_id: int, message: dict):
        if classroom_id in self.active_connections:
            for connection in self.active_connections[classroom_id]:
                await connection.send_json(message)
                
    def get_active_students(self, classroom_id: int) -> list[str]:
        return list(redis_client.smembers(f"classroom:{classroom_id}:students"))

manager = ClassroomManager()
