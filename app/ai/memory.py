from sqlalchemy.orm import Session
from app.models.message import Message

def get_recent_chat_history(db: Session, classroom_id: int, limit: int = 15) -> str:
    messages = db.query(Message).filter(Message.classroom_id == classroom_id).order_by(Message.created_at.desc()).limit(limit).all()
    
    # Reverse to get chronological order
    messages.reverse()
    
    history_str = ""
    for msg in messages:
        sender_name = msg.sender.name if msg.sender else "AI Teacher"
        history_str += f"{sender_name}: {msg.content}\n"
        
    return history_str

def save_message(db: Session, classroom_id: int, sender_id: int | None, sender_type: str, content: str):
    new_message = Message(
        classroom_id=classroom_id,
        sender_id=sender_id,
        sender_type=sender_type,
        content=content
    )
    db.add(new_message)
    db.commit()
    db.refresh(new_message)
    return new_message
