import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from app.ai.graph import classroom_app

def test():
    state = {
        "classroom_id": "test",
        "subject_name": "General",
        "topic_name": "Testing",
        "lecture_date": "Today",
        "active_students": "Shaikh Amir",
        "student_name": "Shaikh Amir",
        "student_profile": "Beginner",
        "chat_history": "No history",
        "question": "Hello teacher",
        "context": "No context",
        "used_analogies": [],
        "strike_count": 0,
        "is_disruptive": False,
        "is_abusive": False
    }
    print("Invoking...")
    result = classroom_app.invoke(state)
    print("Result:")
    print(result.get("chat_content"))
    print(result.get("board_content"))

if __name__ == "__main__":
    test()
