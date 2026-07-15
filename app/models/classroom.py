from sqlalchemy import Column, Integer, String, Enum, DateTime, ForeignKey, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from app.models.base import Base

class ClassroomStatus(str, enum.Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    ENDED = "ENDED"

classroom_students = Table(
    "classroom_students",
    Base.metadata,
    Column("classroom_id", Integer, ForeignKey("classrooms.id")),
    Column("student_id", Integer, ForeignKey("users.id"))
)

class Classroom(Base):
    __tablename__ = "classrooms"

    id = Column(Integer, primary_key=True, index=True)
    teacher_id = Column(Integer, ForeignKey("users.id"))
    topic = Column(String)
    status = Column(Enum(ClassroomStatus), default=ClassroomStatus.PENDING)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    teacher = relationship("User", foreign_keys=[teacher_id])
    students = relationship("User", secondary=classroom_students)
    materials = relationship("Material", back_populates="classroom")
    messages = relationship("Message", back_populates="classroom")


class Material(Base):
    __tablename__ = "materials"
    
    id = Column(Integer, primary_key=True, index=True)
    classroom_id = Column(Integer, ForeignKey("classrooms.id"))
    file_url = Column(String)
    file_type = Column(String) # pdf, ppt, etc.
    processing_status = Column(String, default="PENDING") # PENDING, PROCESSED, ERROR
    
    classroom = relationship("Classroom", back_populates="materials")
