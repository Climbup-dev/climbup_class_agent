from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
import shutil
import os
import uuid
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from langchain_core.documents import Document
from app.core.memory import classroom_brains
from app.core.supabase_client import supabase_new
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
import pickle

load_dotenv()


router = APIRouter()

@router.post("/api/v1/classrooms/{classroom_id}/upload")
async def upload_material(classroom_id: str, file: UploadFile = File(...)):
    allowed_exts = ('.pdf', '.ppt', '.pptx', '.png', '.jpg', '.jpeg')
    if not file.filename.lower().endswith(allowed_exts):
        raise HTTPException(status_code=400, detail="Only PDF, PPT, and Images are supported.")
        
    try:
        supabase_new.storage.create_bucket("class_materials", options={"public": True})
    except Exception:
        pass
    try:
        supabase_new.storage.create_bucket("class_tours", options={"public": True})
    except Exception:
        pass

    try:
        supabase_new.storage.create_bucket("class_materials", options={"public": True})
    except Exception:
        pass
    try:
        supabase_new.storage.create_bucket("class_tours", options={"public": True})
    except Exception:
        pass

    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{classroom_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        with open(file_path, "rb") as f:
            supabase_new.storage.from_("class_materials").upload(f"{classroom_id}.pdf", f.read(), {"content-type": "application/pdf"})
    except Exception:
        pass # Already exists or error
        
    public_pdf_url = supabase_new.storage.from_("class_materials").get_public_url(f"{classroom_id}.pdf")
    try:
        supabase_new.table("classrooms").update({"pdf_url": public_pdf_url}).eq("id", classroom_id).execute()
    except Exception as e:
        print(f"Failed to update db: {e}")
        
    try:
        with open(file_path, "rb") as f:
            supabase_new.storage.from_("class_materials").upload(f"{classroom_id}.pdf", f.read(), {"content-type": "application/pdf"})
    except Exception:
        pass # Already exists or error
        
    public_pdf_url = supabase_new.storage.from_("class_materials").get_public_url(f"{classroom_id}.pdf")
    try:
        supabase_new.table("classrooms").update({"pdf_url": public_pdf_url}).eq("id", classroom_id).execute()
    except Exception as e:
        print(f"Failed to update db: {e}")
        
    try:
        from llama_parse import LlamaParse
        
        # Initialize LlamaParse API
        parser = LlamaParse(
            api_key=os.environ.get("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",  # Extract tables, images, and layout beautifully
            verbose=True
        )
        
        # LlamaParse handles images natively!
        llama_docs = parser.load_data(file_path)
        
        full_text = "\n\n".join([doc.text for doc in llama_docs])
                
        if not full_text.strip():
             raise ValueError("Could not extract any text or image data from the PDF.")
             
        document = Document(page_content=full_text, metadata={"classroom_id": classroom_id, "source": file.filename})
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents([document])
        
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENROUTER_API_KEYS", "dummy").split(",")[0].replace('"', "").replace("'", "").strip()), openai_api_base="https://openrouter.ai/api/v1", model="openai/text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        bm25_retriever = BM25Retriever.from_documents(chunks)
        from app.core.memory import HybridEnsembleRetriever
        faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        bm25_retriever.k = 10
        retriever = HybridEnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.4, 0.6])
        
        classroom_brains[classroom_id] = retriever
        
        # --- GENERATE GUIDED TOUR JSON (One-time, stored permanently) ---
        try:
            import json
            from langchain_groq import ChatGroq
            from langchain_core.prompts import PromptTemplate
            
            tour_prompt = PromptTemplate.from_template("""
            You are a super cool, ultra-smart Indian professor. You have been given the full text of a lecture PDF.
            Your task is to convert this ENTIRE PDF into a highly detailed, structured "Guided Tour" course.
            
            PDF TEXT:
            {full_text}
            
            IMPORTANT RULES:
            1. ONE PAGE = ONE LESSON: The PDF TEXT is clearly divided by page markers (e.g., --- PAGE 1 START ---). You MUST generate exactly ONE single lesson for EVERY SINGLE PAGE. If there are 40 pages in the text, you MUST generate exactly 40 lessons. Combine all paragraphs, concepts, and images on that specific page into that one lesson. Do not skip any page!
            2. EXACT QUOTE (TEXT + IMAGES) - CRITICAL: For "exact_quote", copy and paste the ENTIRE page text verbatim. If the page text contains Markdown images like `![Diagram](URL)`, you MUST NOT delete them! You MUST preserve the exact `![Diagram](URL)` tags in your "exact_quote" output at their original positions. Do NOT summarize or change a single word of the quote. If you strip the images, the system will break!
            3. HINGLISH EXPLANATION: The "explanation" MUST be in a flawless, natural mix of Hindi and English (Hinglish). 
            4. DETAILED & FUN: The explanation should be in FULL DETAIL. Break down the entire grouped concept using funny, real-world, everyday Indian analogies so the student feels "WOW, this is so easy to understand!". Use emojis to make it expressive.
            5. LAYOUT STRUCTURE: The "exact_quote" serves as the TOP section containing the original text and images. The "explanation" serves as the BOTTOM section containing only your Hinglish explanation. DO NOT put images in the explanation!
            
            OUTPUT FORMAT (Strictly valid JSON only, no extra text):
            {{
                "syllabus": ["Topic 1", "Topic 2", "Topic 3", "Topic 4", "...all topics..."],
                "lessons": [
                    {{
                        "topic": "Topic 1",
                        "exact_quote": "The exact verbatim paragraph from the PDF here...",
                        "explanation": "Bhai, isko aise samjho jaise... [Detailed, fun, Hinglish explanation with analogies]"
                    }}
                ]
            }}
            """)
            
            from app.core.llm_balancer import get_balanced_fast_llm
            llm_gemini = get_balanced_fast_llm()
            
            tour_response = llm_gemini.invoke(tour_prompt.format(full_text=full_text))
            
            content = tour_response.content
            if isinstance(content, list):
                content = "".join([str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content])
            
            # Clean markdown wrappers and trailing chars
            content = content.strip()
            if content.startswith("```json"): content = content[7:]
            elif content.startswith("```"): content = content[3:]
            if content.endswith("```"): content = content[:-3]
            content = content.strip()
            
            # Try to validate json
            import json
            try:
                json.loads(content)
            except json.JSONDecodeError:
                if content.endswith("}"):
                    try:
                        content_fixed = content[:-1].strip()
                        json.loads(content_fixed)
                        content = content_fixed
                    except:
                        pass
                        
            tour_json_path = f"uploads/{classroom_id}_tour.json"
            with open(tour_json_path, "w", encoding="utf-8") as tour_file:
                tour_file.write(content)
                
            try:
                with open(tour_json_path, "rb") as f:
                    supabase_new.storage.from_("class_tours").upload(f"{classroom_id}_tour.json", f.read(), {"content-type": "application/json"})
            except Exception:
                pass
                
            import logging
            logging.info(f"✅ Guided Tour generated and saved: {tour_json_path}")
        except Exception as tour_error:
            import logging
            logging.error(f"❌ Tour generation failed: {tour_error}")
        # -----------------------------------------------------------------
            
        return {"status": "success", "message": f"Successfully uploaded and trained AI on {file.filename} using Gemini OCR!"}
    except Exception as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"detail": str(e)})

import threading
faiss_semaphore = threading.Semaphore(1)

def process_upload_in_background(
    file_path: str,
    file_name: str,
    classroom_id: str,
    subject_id: str,
    topic_title: str,
    university_id: str,
    branch_id: str,
    semester_id: str
):
    from llama_parse import LlamaParse
    import os
    import shutil
    import fitz
    import base64
    import logging
    from langchain_core.messages import HumanMessage
    from app.core.llm_balancer import get_balanced_vision_llm, get_balanced_fast_llm
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.retrievers import BM25Retriever
    from app.core.memory import HybridEnsembleRetriever, classroom_brains
    from app.core.supabase_client import supabase_new
    import pickle
    import json
    from langchain_core.prompts import PromptTemplate
    import uuid
    
    try:
        bucket_name = "class_materials"
        parser = LlamaParse(
            api_key=os.environ.get("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            verbose=True
        )
        
        logging.info("Starting Extraction using LlamaParse...")
        llama_docs = parser.load_data(file_path)
        
        # --- SIMPLIFIED EXTRACTION (LlamaParse Text Only) ---
        full_text = "\n\n".join([doc.text.strip() for doc in llama_docs])
        
        if not full_text.strip():
             raise ValueError("Could not extract any text from the file.")
        # ---------------------------------
             
        metadata = {
            "classroom_id": classroom_id, 
            "university_id": university_id,
            "branch_id": branch_id,
            "semester_id": semester_id,
            "subject_id": subject_id,
            "topic_title": topic_title,
            "source": file_name
        }
        
        # USE SEMAPHORE TO PREVENT OOM CRASHES DURING CONCURRENT FAISS BUILDS
        with faiss_semaphore:
            logging.info(f"Acquired RAM lock for building FAISS for {classroom_id}")
            document = Document(page_content=full_text, metadata=metadata)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents([document])
            
            embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENROUTER_API_KEYS", "dummy").split(",")[0].replace('"', "").replace("'", "").strip()), openai_api_base="https://openrouter.ai/api/v1", model="openai/text-embedding-3-small")
            vector_store = FAISS.from_documents(chunks, embeddings)
            
            bm25_retriever = BM25Retriever.from_documents(chunks)
            faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
            bm25_retriever.k = 10
            retriever = HybridEnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.4, 0.6])
            
            classroom_brains[classroom_id] = retriever
            
            # --- FAISS Cloud Backup ---
            faiss_dir = f"faiss_{classroom_id}"
            vector_store.save_local(faiss_dir)
        
        with open(os.path.join(faiss_dir, "bm25.pkl"), "wb") as f:
            pickle.dump(bm25_retriever, f)
        
        zip_name = f"{faiss_dir}.zip"
        shutil.make_archive(faiss_dir, 'zip', faiss_dir)
        
        bucket_vector = "vector_stores"
        try:
            supabase_new.storage.create_bucket(bucket_vector, options={"public": False})
        except Exception:
            pass # Bucket exists
            
        with open(zip_name, "rb") as f:
            supabase_new.storage.from_(bucket_vector).upload(zip_name, f.read(), {"content-type": "application/zip"})
            
        os.remove(zip_name)
        shutil.rmtree(faiss_dir)
        logging.info(f"Background processing complete for {classroom_id}")
        
        # Notify clients that course is ready
        try:
            import json
            import asyncio
            from app.api.websockets import manager
            # We need to run the async broadcast in the event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(manager.broadcast_to_classroom(classroom_id, json.dumps({"type": "course_ready", "classroom_id": classroom_id})))
            logging.info("Broadcasted course_ready to frontend!")
        except Exception as e:
            logging.warning(f"Failed to broadcast course_ready: {e}")
            
    except Exception as e:
        logging.error(f"Background processing failed: {e}")

@router.post("/api/v1/upload-smart")
async def upload_smart_material(
    background_tasks: BackgroundTasks,
    university_id: str = Form(...),
    branch_id: str = Form(...),
    semester_id: str = Form(...),
    subject_id: str = Form(...),
    topic_title: str = Form(...),
    file: UploadFile = File(...)
):
    import uuid
    import shutil
    import os
    from app.core.supabase_client import supabase_new
    
    allowed_exts = ('.pdf', '.ppt', '.pptx', '.png', '.jpg', '.jpeg')
    if not file.filename.lower().endswith(allowed_exts):
        raise HTTPException(status_code=400, detail="Only PDF, PPT, and Images are supported.")
        
    classroom_id = str(uuid.uuid4())
    
    try:
        supabase_new.storage.create_bucket("class_materials", options={"public": True})
    except Exception:
        pass
    try:
        supabase_new.storage.create_bucket("class_tours", options={"public": True})
    except Exception:
        pass

    try:
        supabase_new.storage.create_bucket("class_materials", options={"public": True})
    except Exception:
        pass
    try:
        supabase_new.storage.create_bucket("class_tours", options={"public": True})
    except Exception:
        pass

    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{classroom_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
        # Removed duplicate upload blocks
        
    try:
        supabase_new.table('subjects').upsert({
            "id": subject_id, 
            "subject_name": f"Subject {subject_id}"
        }).execute()
        
        # Upload to Supabase Storage
        bucket_name = "class_materials"
        try:
            supabase_new.storage.create_bucket(bucket_name, options={"public": True})
        except Exception:
            pass # Bucket likely already exists
            
        storage_path = f"{classroom_id}.pdf"
        with open(file_path, "rb") as f:
            supabase_new.storage.from_(bucket_name).upload(storage_path, f.read(), {"content-type": "application/pdf"})
            
        pdf_url = supabase_new.storage.from_(bucket_name).get_public_url(storage_path)

        session_data = {
            "id": classroom_id,
            "subject_id": subject_id,
            "topic_name": topic_title,
            "pdf_url": pdf_url
        }
        supabase_new.table('classrooms').insert(session_data).execute()

        # Add background task for heavy processing
        background_tasks.add_task(
            process_upload_in_background,
            file_path=file_path,
            file_name=file.filename,
            classroom_id=classroom_id,
            subject_id=subject_id,
            topic_title=topic_title,
            university_id=university_id,
            branch_id=branch_id,
            semester_id=semester_id
        )
            
        return {
            "status": "processing", 
            "classroom_id": classroom_id,
            "message": f"Successfully uploaded {file.filename}. The AI is processing the course in the background."
        }
    except Exception as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"detail": str(e)})
