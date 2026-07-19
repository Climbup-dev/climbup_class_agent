from app.core.llm_balancer import get_balanced_text_llm
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
from langchain_core.prompts import PromptTemplate
from app.core.memory import classroom_brains
import os
import datetime
import pytz
from langchain_core.prompts import PromptTemplate
from app.core.memory import classroom_brains
from app.core.supabase_client import supabase_new
from dotenv import load_dotenv
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ai.graph import classroom_app
from app.rag.embedder import reformulate_query, rerank_documents
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
        # Multi-Agent Memory
        self.used_analogies: Dict[str, list] = {}
        # Moderation Tracking
        self.strikes: Dict[str, Dict[str, int]] = {}

    async def connect(self, websocket: WebSocket, classroom_id: str, student_id: str, student_name: str):
        if classroom_id not in self.active_connections:
            self.active_connections[classroom_id] = []
            self.strikes[classroom_id] = {}
        self.active_connections[classroom_id].append(websocket)
        self.connection_names[websocket] = student_name

        # 1. Fetch Session and Subject Info from Supabase (if not cached)
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
            
            # Record entry in session_students if not exists
            supabase_new.table('classroom_students').upsert({
                "classroom_id": classroom_id,
                "student_id": student_id
            }).execute()
        except Exception as e:
            pass # Ignore errors for MVP if DB isn't perfectly set up yet

        self.llm = get_balanced_text_llm(temperature=0.7)
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENROUTER_API_KEY', os.environ.get('OPENROUTER_API_KEYS', 'dummy').split(',')[0].replace('"', '').replace("'", "").strip()), openai_api_base="https://openrouter.ai/api/v1", model="openai/text-embedding-3-small")
    def disconnect(self, websocket: WebSocket, classroom_id: str):
        if classroom_id in self.active_connections:
            self.active_connections[classroom_id].remove(websocket)
            # Clear chat history when the classroom is empty
            if len(self.active_connections[classroom_id]) == 0:
                if classroom_id in self.classroom_history:
                    del self.classroom_history[classroom_id]
        if websocket in self.connection_names:
            del self.connection_names[websocket]
            
    def get_active_students(self, classroom_id: str) -> List[str]:
        if classroom_id not in self.active_connections:
            return []
        return list(set(self.connection_names[ws] for ws in self.active_connections[classroom_id] if ws in self.connection_names))

    async def broadcast(self, message: dict, classroom_id: str, exclude: WebSocket = None):
        if classroom_id in self.active_connections:
            for connection in self.active_connections[classroom_id]:
                if exclude and connection == exclude:
                    continue
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

manager = ConnectionManager()

prompt_template = "" # Removed, now handled by LangGraph in app.ai.graph

@router.websocket("/ws/classroom/{classroom_id}")
async def websocket_endpoint(websocket: WebSocket, classroom_id: str, student_id: str, student_name: str):
    await websocket.accept()
    
    # --- TIME RESTRICTION LOGIC (TEMPORARILY DISABLED FOR TESTING) ---
    # ist = pytz.timezone('Asia/Kolkata')
    # current_time = datetime.datetime.now(ist).time()
    # 
    # # Define Class Hours (8:00 PM to 9:00 PM IST)
    # start_time = datetime.time(20, 0)
    # end_time = datetime.time(21, 0)
    # 
    # if not (start_time <= current_time <= end_time):
    #     await websocket.send_json({
    #         "type": "error",
    #         "content": "Classroom is Closed! ΓÅ│ Live group sessions are only open between 8:00 PM and 9:00 PM. Please use the Personal Chatbot for 24/7 help."
    #     })
    #     await websocket.close()
    #     return
    # ------------------------------------------------------------------
    
    await manager.connect(websocket, classroom_id, student_id, student_name)
    
    try:
        if classroom_id not in manager.classroom_history:
            manager.classroom_history[classroom_id] = []

        # --- FAISS Lazy Load & Initial Overview ---
        try:
            vector_store = classroom_brains.get(classroom_id)
            if not vector_store:
                try:
                    bucket_vector = "vector_stores"
                    zip_name = f"faiss_{classroom_id}.zip"
                    faiss_dir = f"faiss_{classroom_id}"
                    
                    res = supabase_new.storage.from_(bucket_vector).download(zip_name)
                    with open(zip_name, "wb") as f:
                        f.write(res)
                        
                    import shutil
                    import os
                    shutil.unpack_archive(zip_name, faiss_dir)
                    
                    from langchain_community.vectorstores import FAISS
                    from app.core.memory import HybridEnsembleRetriever
                    import pickle
                    
                    faiss_vs = FAISS.load_local(faiss_dir, manager.embeddings, allow_dangerous_deserialization=True)
                    faiss_retriever = faiss_vs.as_retriever(search_kwargs={"k": 10})
                    
                    bm25_path = os.path.join(faiss_dir, "bm25.pkl")
                    if os.path.exists(bm25_path):
                        with open(bm25_path, "rb") as f:
                            bm25_retriever = pickle.load(f)
                        bm25_retriever.k = 10
                        retriever = HybridEnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.4, 0.6])
                    else:
                        retriever = faiss_retriever
                        
                    classroom_brains[classroom_id] = retriever
                    
                    os.remove(zip_name)
                    shutil.rmtree(faiss_dir)
                except Exception as faiss_err:
                    print("Could not lazy load FAISS:", faiss_err)
                    vector_store = None
            
            retriever = classroom_brains.get(classroom_id)
            if retriever:
                try:
                    docs = retriever.invoke("syllabus overview important topics concepts")[:4]
                    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
                except Exception as e:
                    pass
                ctx = manager.classroom_contexts.get(classroom_id, {"subject_name": "General Topic", "topic_name": "General Lecture"})
                prof = manager.student_profiles.get(student_id, {"engagement_level": "Unknown"})
                
                # We no longer trigger a SYSTEM_INIT AI response because it overrides the Course Tour UI.
                # Do not send any system message here to avoid spamming the UI.
                
        except Exception as init_err:
            print("Init Error:", init_err)
        # ------------------------------------------

        while True:
            data = await websocket.receive_text()
            
            manager.classroom_history[classroom_id].append(f"{student_name}: {data}")
            
            try:
                # Update engagement stats
                prof = manager.student_profiles.get(student_id, {})
                if prof:
                    prof['total_messages_sent'] = prof.get('total_messages_sent', 0) + 1
                    supabase_new.table('student_profiles').update({
                        "total_messages_sent": prof['total_messages_sent']
                    }).eq('student_id', student_id).execute()
            except Exception as e:
                pass
            # Echo the user's message back to themselves (optional, usually handled by frontend, but we'll send a confirmation if needed. For now, we skip echoing to avoid duplicates on the frontend if frontend already shows it).
            # await websocket.send_json({
            #     "type": "chat",
            #     "sender": student_name,
            #     "content": data
            # })
            
            try:
                retriever = classroom_brains.get(classroom_id)

                if retriever:
                    history = manager.classroom_history.get(classroom_id, [])[-5:-1]
                    chat_history_str = "\n".join(history) if history else "No previous history."
                    
                    context = ""
                else:
                    context = "No specific lecture material found. Please ask the teacher to upload a PDF."
                
                history = manager.classroom_history[classroom_id][-5:-1]
                chat_history_str = "\n".join(history) if history else "No previous history."
                
                active_students = ", ".join(manager.get_active_students(classroom_id))
                
                ctx = manager.classroom_contexts.get(classroom_id, {"subject_name": "General Topic", "topic_name": "General Lecture"})
                prof = manager.student_profiles.get(student_id, {"engagement_level": "Unknown"})
                
                used_analogies = manager.used_analogies.get(classroom_id, [])
                current_strikes = manager.strikes.get(classroom_id, {}).get(student_id, 0)
                
                state = {
                    "retriever": retriever,
                    "classroom_id": classroom_id,
                    "subject_name": ctx["subject_name"],
                    "topic_name": ctx["topic_name"],
                    "lecture_date": ctx.get("created_at", "Today"),
                    "active_students": active_students,
                    "student_name": student_name,
                    "student_profile": f"Engagement Level: {prof.get('engagement_level', 'Unknown')}, Total Messages: {prof.get('total_messages_sent', 0)}",
                    "chat_history": chat_history_str,
                    "question": data,
                    "context": context,
                    "used_analogies": used_analogies,
                    "strike_count": current_strikes,
                    "is_disruptive": False,
                    "is_abusive": False
                }
                
                # Call the Multi-Agent Pipeline
                try:
                    result = await classroom_app.ainvoke(state)
                    manager.used_analogies[classroom_id] = result.get("used_analogies", [])
                    
                    board_content = result.get("board_content", "")
                    chat_content = result.get("chat_content", "")
                    
                    # Check Moderation Output from the Router Node
                    is_disruptive = result.get("is_disruptive", False)
                    is_abusive = result.get("is_abusive", False)
                    
                    # No kicking for personal sessions, let the Persona scold them!
                            
                    # Gamification: Check if XP was awarded
                    awarded_xp = result.get("awarded_xp", 0)
                    if awarded_xp > 0:
                        try:
                            # Safely fetch current XP from Supabase to increment it
                            # xp_res = supabase_new.table('student_profiles').select('xp_points').eq('student_id', student_id).execute()
                            # current_xp = xp_res.data[0].get('xp_points', 0) if xp_res.data else 0
                            # new_xp = current_xp + awarded_xp
                            
                            # Update in DB
                            # supabase_new.table('student_profiles').update({"xp_points": new_xp}).eq('student_id', student_id).execute()
                            
                            # Send award event directly to the user (frontend can handle visual without DB update for now)
                            await websocket.send_json({
                                "type": "award",
                                "student": student_name,
                                "points": awarded_xp,
                                "content": f"≡ƒÄë You earned {awarded_xp} XP for a brilliant answer!"
                            })
                        except Exception as xp_err:
                            print("Error updating XP:", xp_err)
                            
                except Exception as e:
                    print("Multi-Agent Error:", e)
                    chat_content = "SILENCE"
                    board_content = ""
                
                # If AI decides not to speak, do nothing.
                if chat_content.upper() == "SILENCE" or "SILENCE" in chat_content[:10].upper():
                    if not board_content:
                        continue
                
                manager.classroom_history[classroom_id].append(f"AI Teacher: {chat_content}")

            except Exception as e:
                chat_content = f"I'm sorry, my AI brain encountered an error: {str(e)}"
                board_content = ""

            payload = {
                "type": "teaching",
                "sender": "AI Teacher",
                "chat_content": chat_content,
                "board_content": board_content
            }
            print(f"DEBUG SENDING WEBSOCKET: {payload}")
            
            await websocket.send_json(payload)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, classroom_id)
