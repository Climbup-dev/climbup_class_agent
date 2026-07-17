import litellm
from sqlalchemy.orm import Session
from app.rag.embedder import retrieve_context
from app.ai.prompts import get_master_agent_prompt
from app.ai.memory import get_recent_chat_history
from app.models.classroom import Classroom
from app.core.config import settings
import os

# Set API keys for LiteLLM
os.environ["OPENROUTER_API_KEY"] = settings.OPENROUTER_API_KEY
os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY

def generate_classroom_response(
    db: Session, 
    classroom_id: int, 
    student_query: str, 
    active_students: list[str],
    model: str = "groq/llama3-70b-8192" # Defaulting to Groq for speed, can be gemini/gemini-1.5-pro or openrouter/...
) -> str:
    
    # 1. Fetch Classroom details
    classroom = db.query(Classroom).filter(Classroom.id == classroom_id).first()
    if not classroom:
        return "Error: Classroom not found."
        
    topic = classroom.topic or "General Lecture"
    
    # 2. Get Recent Chat History FIRST so we can use it for Query Reformulation
    chat_history = get_recent_chat_history(db, classroom_id, limit=10)
    
    # 3. Retrieve Context via Optimized RAG (passes chat history)
    retrieved_context = retrieve_context(query=student_query, classroom_id=classroom_id, chat_history=chat_history, top_k=3)
    if not retrieved_context.strip():
        retrieved_context = "No specific context found from uploaded materials."
    
    # 4. Construct System Prompt
    system_prompt = get_master_agent_prompt(topic=topic, active_students=active_students)
    
    # 5. Construct User Prompt
    user_prompt = f"""
RETRIEVED LECTURE CONTEXT:
{retrieved_context}

RECENT CHAT HISTORY:
{chat_history}

NEW STUDENT MESSAGE:
{student_query}

Craft your response as the AI Teacher based on the rules provided.
"""

    # 6. Call LLM via LiteLLM
    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating AI response: {e}")
        return "I'm having a little trouble thinking right now. Could you repeat that?"
