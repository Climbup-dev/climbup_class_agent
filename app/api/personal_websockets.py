from app.core.llm_balancer import get_balanced_text_llm
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
import os
from langchain_core.prompts import PromptTemplate
from app.core.memory import classroom_brains
from app.core.supabase_client import supabase_new

load_dotenv()

router = APIRouter()

class PersonalConnectionManager:
    def __init__(self):
        # We map student_id -> list of websockets (in case of multiple tabs)
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # History will be stored per student_id for 1-on-1 isolation
        self.personal_history: Dict[str, List[str]] = {}
        
        # Classroom Context Cache
        self.classroom_contexts: Dict[str, dict] = {}
        # Student Profile Cache
        self.student_profiles: Dict[str, dict] = {}

    async def connect(self, websocket: WebSocket, classroom_id: str, student_id: str, student_name: str):
        await websocket.accept()
        if student_id not in self.active_connections:
            self.active_connections[student_id] = []
        self.active_connections[student_id].append(websocket)

        # 1. Fetch Session and Subject Info from Supabase
        if classroom_id not in self.classroom_contexts:
            try:
                res = supabase_new.table('classrooms').select('topic_name, subject_id, created_at').eq('id', classroom_id).execute()
                if res.data:
                    c_data = res.data[0]
                    sub_res = supabase_new.table('subjects').select('subject_name').eq('id', c_data['subject_id']).execute()
                    subject_name = sub_res.data[0]['subject_name'] if sub_res.data else "General Topic"
                    self.classroom_contexts[classroom_id] = {
                        "subject_name": subject_name,
                        "topic_name": c_data['topic_name'],
                        "lecture_date": c_data.get('created_at', 'Today')
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
            
            supabase_new.table('classroom_students').upsert({
                "session_id": classroom_id,
                "student_id": student_id
            }).execute()
        except Exception as e:
            pass 

        # Lower temperature for systematic, accurate answers
        self.llm = get_balanced_text_llm(temperature=0.2)
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENROUTER_API_KEY', os.environ.get('OPENROUTER_API_KEYS', 'dummy').split(',')[0].replace('"', '').replace("'", "").strip()), openai_api_base="https://openrouter.ai/api/v1", model="openai/text-embedding-3-small")

    def disconnect(self, websocket: WebSocket, student_id: str):
        if student_id in self.active_connections:
            if websocket in self.active_connections[student_id]:
                self.active_connections[student_id].remove(websocket)
            if not self.active_connections[student_id]:
                del self.active_connections[student_id]

    async def send_personal(self, message: dict, student_id: str):
        if student_id in self.active_connections:
            for connection in self.active_connections[student_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

personal_manager = PersonalConnectionManager()

personal_prompt_template = """
You are an Expert, Serious, and highly Systematic Personal Tutor AI. 
You are currently providing 1-on-1 personalized tutoring for {student_name} on the subject of "{subject_name}".
The specific topic of focus today is "{topic_name}".

Your goal is to provide deep, accurate, and highly structured explanations. 
Unlike a casual group chat, here you MUST:
1. Act like a top-tier professor: Break down complex concepts into simple, logical steps.
2. Rely heavily on the provided RAG Context. Do not hallucinate. If the context doesn't have the exact answer, use your expert knowledge but clarify it.
3. Use formatting (bullet points, numbered lists, bold text) to make your explanations systematic and easy to digest.
4. Encourage the student to think deeply by ending with a thought-provoking question related to the concept.
5. NEVER use slang, roasting, or inappropriate humor. Maintain a professional, highly encouraging, and academic tone.

Student Profile: {student_profile}

Recent Personal Conversation History:
{chat_history}

Context from Course Material:
{context}

Student Question: {question}
Systematic Answer:
"""

@router.websocket("/ws/personal/{classroom_id}")
async def personal_websocket_endpoint(websocket: WebSocket, classroom_id: str, student_id: str, student_name: str):
    await personal_manager.connect(websocket, classroom_id, student_id, student_name)
    
    # We do not broadcast to everyone, only to this specific student's tabs
    await personal_manager.send_personal({
        "type": "system",
        "content": f"Personal Tutor Mode Activated. Ready to dive deep into today's concepts."
    }, student_id)
    
    try:
        if student_id not in personal_manager.personal_history:
            personal_manager.personal_history[student_id] = []

        while True:
            data = await websocket.receive_text()
            
            personal_manager.personal_history[student_id].append(f"{student_name}: {data}")
            
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
                prof = personal_manager.student_profiles.get(student_id, {})
                if prof:
                    prof['total_messages_sent'] = prof.get('total_messages_sent', 0) + 1
                    supabase_new.table('student_profiles').update({
                        "total_messages_sent": prof['total_messages_sent']
                    }).eq('student_id', student_id).execute()
            except Exception as e:
                pass
            
            # Echo back what they typed
            await personal_manager.send_personal({
                "type": "chat",
                "sender": student_name,
                "content": data
            }, student_id)
            
            try:
                retriever = classroom_brains.get(classroom_id)
                if retriever:
                    # K=5 for deeper search in personal mode
                    docs = retriever.invoke(data)
                    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
                else:
                    context = "No specific lecture material found. Rely on your general expert knowledge."
                
                # Retrieve history for THIS specific student
                history = personal_manager.personal_history[student_id][-10:-1]
                chat_history_str = "\n".join(history) if history else "No previous history."
                
                ctx = personal_manager.classroom_contexts.get(classroom_id, {"subject_name": "General Subject", "topic_name": "Core Concepts"})
                prof = personal_manager.student_profiles.get(student_id, {"engagement_level": "Unknown"})
                
                prompt = PromptTemplate.from_template(personal_prompt_template).format(
                    subject_name=ctx["subject_name"],
                    topic_name=ctx["topic_name"],
                    student_name=student_name,
                    student_profile=f"Engagement Level: {prof.get('engagement_level', 'Unknown')}, Total Messages: {prof.get('total_messages_sent', 0)}",
                    context=context,
                    chat_history=chat_history_str,
                    question=data
                )
                
                ai_response = await personal_manager.llm.ainvoke(prompt)
                response_text = ai_response.content.strip()
                
                personal_manager.personal_history[student_id].append(f"Personal Tutor AI: {response_text}")
                
                # Save AI message to Supabase
                try:
                    supabase_new.table('messages').insert({
                        "session_id": classroom_id,
                        "sender_id": None,
                        "sender_type": "ai",
                        "sender_name": "Personal Tutor AI",
                        "content": response_text
                    }).execute()
                except Exception as e:
                    pass

            except Exception as e:
                response_text = f"System error occurred while generating explanation: {str(e)}"

            await personal_manager.send_personal({
                "type": "chat",
                "sender": "Personal Tutor AI",
                "content": response_text
            }, student_id)
            
    except WebSocketDisconnect:
        personal_manager.disconnect(websocket, student_id)
