from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.models.base import Base

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    classroom_id = Column(Integer, ForeignKey("classrooms.id"))
    sender_id = Column(Integer, ForeignKey("users.id"), nullable=True) # Null if AI
    sender_type = Column(String) # "USER" or "AI"
    content = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    classroom = relationship("Classroom", back_populates="messages")
    sender = relationship("User")
