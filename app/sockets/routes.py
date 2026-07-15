from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
from app.sockets.manager import manager
from app.ai.agent import generate_classroom_response
from app.ai.memory import save_message
from app.core.database import get_db

router = APIRouter()

@router.websocket("/ws/classroom/{classroom_id}")
async def classroom_websocket(
    websocket: WebSocket, 
    classroom_id: int, 
    student_id: int, 
    student_name: str,
    db: Session = Depends(get_db)
):
    await manager.connect(websocket, classroom_id, student_name)
    
    # Broadcast student joined
    await manager.broadcast_to_classroom(classroom_id, {
        "type": "system",
        "content": f"{student_name} joined the classroom."
    })
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # 1. Save student message to DB
            save_message(db, classroom_id, student_id, "STUDENT", data)
            
            # 2. Broadcast student message to everyone in the room
            await manager.broadcast_to_classroom(classroom_id, {
                "type": "chat",
                "sender": student_name,
                "content": data
            })
            
            # 3. Trigger AI Agent
            active_students = manager.get_active_students(classroom_id)
            
            # (In production, this should be an async background task to prevent blocking the WS)
            ai_response = generate_classroom_response(
                db=db, 
                classroom_id=classroom_id, 
                student_query=data, 
                active_students=active_students
            )
            
            # 4. Save AI message to DB
            save_message(db, classroom_id, None, "AI", ai_response)
            
            # 5. Broadcast AI response
            await manager.broadcast_to_classroom(classroom_id, {
                "type": "chat",
                "sender": "AI Teacher",
                "content": ai_response
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, classroom_id, student_name)
        await manager.broadcast_to_classroom(classroom_id, {
            "type": "system",
            "content": f"{student_name} left the classroom."
        })
