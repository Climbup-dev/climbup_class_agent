from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from app.core.memory import classroom_brains
from app.core.supabase_client import supabase_new

load_dotenv()

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.classroom_history: Dict[str, List[str]] = {}
        self.connection_names: Dict[WebSocket, str] = {}
        
        # Classroom Context Cache
        self.classroom_contexts: Dict[str, dict] = {}
        # Student Profile Cache
        self.student_profiles: Dict[str, dict] = {}

    async def connect(self, websocket: WebSocket, classroom_id: str, student_id: str, student_name: str):
        await websocket.accept()
        if classroom_id not in self.active_connections:
            self.active_connections[classroom_id] = []
        self.active_connections[classroom_id].append(websocket)
        self.connection_names[websocket] = student_name

        # 1. Fetch Session and Subject Info from Supabase (if not cached)
        if classroom_id not in self.classroom_contexts:
            try:
                res = supabase_new.table('classrooms').select('topic_name, subject_id, lecture_date').eq('id', classroom_id).execute()
                if res.data:
                    c_data = res.data[0]
                    sub_res = supabase_new.table('subjects').select('subject_name').eq('id', c_data['subject_id']).execute()
                    subject_name = sub_res.data[0]['subject_name'] if sub_res.data else "General Topic"
                    self.classroom_contexts[classroom_id] = {
                        "subject_name": subject_name,
                        "topic_name": c_data['topic_name'],
                        "lecture_date": c_data.get('lecture_date', 'Today')
                    }
                else:
                    self.classroom_contexts[classroom_id] = {"subject_name": "General Topic", "topic_name": "General Lecture", "lecture_date": "Today"}
            except Exception as e:
                self.classroom_contexts[classroom_id] = {"subject_name": "General Topic", "topic_name": "General Lecture", "lecture_date": "Today"}

        # 2. Fetch or Create Student Profile
        try:
            prof_res = supabase_new.table('student_profiles').select('*').eq('student_id', student_id).execute()
            if not prof_res.data:
                new_prof = {
                    "student_id": student_id,
                    "student_name": student_name,
                    "total_messages_sent": 0,
                    "engagement_level": "Beginner"
                }
                supabase_new.table('student_profiles').insert(new_prof).execute()
                self.student_profiles[student_id] = new_prof
            else:
                self.student_profiles[student_id] = prof_res.data[0]
            
            # Record entry in session_students if not exists
            supabase_new.table('session_students').upsert({
                "session_id": classroom_id,
                "student_id": student_id
            }).execute()
        except Exception as e:
            pass # Ignore errors for MVP if DB isn't perfectly set up yet

        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))

    def disconnect(self, websocket: WebSocket, classroom_id: str):
        if classroom_id in self.active_connections:
            self.active_connections[classroom_id].remove(websocket)
        if websocket in self.connection_names:
            del self.connection_names[websocket]
            
    def get_active_students(self, classroom_id: str) -> List[str]:
        if classroom_id not in self.active_connections:
            return []
        return list(set(self.connection_names[ws] for ws in self.active_connections[classroom_id] if ws in self.connection_names))

    async def broadcast(self, message: dict, classroom_id: str):
        if classroom_id in self.active_connections:
            for connection in self.active_connections[classroom_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

manager = ConnectionManager()

prompt_template = """
You are "Humor Engine", a highly intelligent, witty, and sarcastic AI Mentor in a group chat. You are NOT a traditional teacher. You do NOT give boring lectures. 
You are specifically hired to teach the subject "{subject_name}". 
Today's Date is {lecture_date} and the focus topic for today's session is "{topic_name}".
Rule: ALL your games, analogies, and roasting MUST be strictly themed around {subject_name}!

Your goal is to "play" with the topic and enjoy with the students. You must make the concepts reach their subconscious mind by turning the topic into fun, interactive mini-games, playful challenges, or mind-bending riddles.
You teach using a natural mix of Hinglish (Hindi + English). You treat students like friends, playfully roasting them, cracking spontaneous natural jokes, and gamifying the knowledge.

Core Persona Rules:
1. SMART INTERVENTION: You DO NOT need to reply to every small casual message (like "hi", "ok", or quick banter). If they don't need you, reply with the exact word "SILENCE". 
HOWEVER, you MUST intervene and reply if:
- They ask a question or seem stuck.
- They drift completely off-topic. (Playfully roast them and challenge them with a fun, topic-related mini-game to drag them back).
2. PLAY GAMES WITH THE TOPIC: Do not just "explain" things. Gamify it! Give them situations, ask them to guess, create a small rapid-fire challenge, or a roleplay scenario based on the topic. The knowledge must stick in their subconscious mind through PLAY.
3. MULTIPLAYER AWARENESS: You are managing a live group of friends/students. You know exactly who is talking. Address them by name.
4. STUDENT PROFILE AWARENESS: You have access to the student's engagement profile. Adjust your tone accordingly! If they are 'Quiet', encourage them playfully. If they are a 'Class Clown', roast them back!
5. REAL WORLD "WOW" EXAMPLES: When you do reveal the answer, use mind-blowing, highly relatable everyday examples that make them say "WOW!".
6. REAL HUMAN VIBE & MEMORY: You have perfect memory of the conversation. Refer back to what students said earlier naturally. Be extremely intelligent and spontaneous.
7. NO REPETITION: Create YOUR OWN fresh, original analogies and games based on the CURRENT topic. NEVER repeat old examples (no samosa/packet jokes, those are banned).
8. NO SENSITIVE TOPICS: No politics, religion, caste, race, or personal appearance jokes.

CRITICAL FORMATTING:
- If you decide NOT to intervene, output exactly and only: SILENCE
- If you DO speak, keep your responses EXTREMELY SHORT, punchy, and conversational (Max 3-4 sentences). 

Currently Active Students in Chat: {active_students}

Recent Conversation History:
{chat_history}

Context from Material:
{context}

Current Student Speaking: {student_name}
Student Profile Data: {student_profile}
Question/Message: {question}
Answer:
"""

@router.websocket("/ws/classroom/{classroom_id}")
async def websocket_endpoint(websocket: WebSocket, classroom_id: str, student_id: str, student_name: str):
    await manager.connect(websocket, classroom_id, student_id, student_name)
    
    await manager.broadcast({
        "type": "system",
        "content": f"{student_name} joined the classroom."
    }, classroom_id)
    
    try:
        if classroom_id not in manager.classroom_history:
            manager.classroom_history[classroom_id] = []

        while True:
            data = await websocket.receive_text()
            
            manager.classroom_history[classroom_id].append(f"{student_name}: {data}")
            
            # Save student message to Supabase
            try:
                supabase_new.table('messages').insert({
                    "session_id": classroom_id,
                    "sender_id": student_id,
                    "sender_type": "student",
                    "sender_name": student_name,
                    "content": data
                }).execute()
                
                # Update engagement stats
                prof = manager.student_profiles.get(student_id, {})
                if prof:
                    prof['total_messages_sent'] = prof.get('total_messages_sent', 0) + 1
                    supabase_new.table('student_profiles').update({
                        "total_messages_sent": prof['total_messages_sent']
                    }).eq('student_id', student_id).execute()
            except Exception as e:
                pass
            
            await manager.broadcast({
                "type": "chat",
                "sender": student_name,
                "content": data
            }, classroom_id)
            
            try:
                vector_store = classroom_brains.get(classroom_id)
                if vector_store:
                    docs = vector_store.similarity_search(data, k=3)
                    context = "\n".join([doc.page_content for doc in docs])
                else:
                    context = "No specific lecture material found. Please ask the teacher to upload a PDF."
                
                history = manager.classroom_history[classroom_id][-15:-1]
                chat_history_str = "\n".join(history) if history else "No previous history."
                
                active_students = ", ".join(manager.get_active_students(classroom_id))
                
                ctx = manager.classroom_contexts.get(classroom_id, {"subject_name": "General Topic", "topic_name": "General Lecture"})
                prof = manager.student_profiles.get(student_id, {"engagement_level": "Unknown"})
                
                prompt = PromptTemplate.from_template(prompt_template).format(
                    subject_name=ctx["subject_name"],
                    topic_name=ctx["topic_name"],
                    lecture_date=ctx.get("lecture_date", "Today"),
                    active_students=active_students,
                    student_name=student_name,
                    student_profile=f"Engagement Level: {prof.get('engagement_level', 'Unknown')}, Total Messages: {prof.get('total_messages_sent', 0)}",
                    context=context,
                    chat_history=chat_history_str,
                    question=data
                )
                
                ai_response = await manager.llm.ainvoke(prompt)
                response_text = ai_response.content.strip()
                
                # If AI decides not to speak, do nothing.
                if response_text.upper() == "SILENCE" or "SILENCE" in response_text[:10].upper():
                    continue
                
                manager.classroom_history[classroom_id].append(f"AI Teacher: {response_text}")
                
                # Save AI message to Supabase
                try:
                    supabase_new.table('messages').insert({
                        "session_id": classroom_id,
                        "sender_id": None,
                        "sender_type": "ai",
                        "sender_name": "AI Teacher",
                        "content": response_text
                    }).execute()
                except Exception as e:
                    pass

            except Exception as e:
                response_text = f"I'm sorry, my AI brain encountered an error: {str(e)}"

            await manager.broadcast({
                "type": "chat",
                "sender": "AI Teacher",
                "content": response_text
            }, classroom_id)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, classroom_id)
        await manager.broadcast({
            "type": "system",
            "content": f"{student_name} left the classroom."
        }, classroom_id)
