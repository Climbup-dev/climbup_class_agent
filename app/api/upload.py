from fastapi import APIRouter, UploadFile, File, Form, HTTPException
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

load_dotenv()


router = APIRouter()

@router.post("/api/v1/classrooms/{classroom_id}/upload")
async def upload_material(classroom_id: str, file: UploadFile = File(...)):
    allowed_exts = ('.pdf', '.ppt', '.pptx', '.png', '.jpg', '.jpeg')
    if not file.filename.lower().endswith(allowed_exts):
        raise HTTPException(status_code=400, detail="Only PDF, PPT, and Images are supported.")
        
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{classroom_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
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
        
        formatted_pages = []
        for i, doc in enumerate(llama_docs):
            page_num = i + 1
            formatted_pages.append(f"[--- PAGE {page_num} START ---]\n{doc.text}\n[--- PAGE {page_num} END ---]")
            
        full_text = "\n\n".join(formatted_pages)
                
        if not full_text.strip():
             raise ValueError("Could not extract any text or image data from the PDF.")
             
        # Save directly to memory (No FAISS)
        classroom_brains[classroom_id] = full_text
            
        return {"status": "success", "message": f"Successfully uploaded and trained AI on {file.filename} using Gemini OCR!"}
    except Exception as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"detail": str(e)})

@router.post("/api/v1/upload-smart")
async def upload_smart_material(
    university_id: str = Form(...),
    branch_id: str = Form(...),
    semester_id: str = Form(...),
    subject_id: str = Form(...),
    topic_title: str = Form(...),
    file: UploadFile = File(...)
):
    allowed_exts = ('.pdf', '.ppt', '.pptx', '.png', '.jpg', '.jpeg')
    if not file.filename.lower().endswith(allowed_exts):
        raise HTTPException(status_code=400, detail="Only PDF, PPT, and Images are supported.")
        
    classroom_id = str(uuid.uuid4())
    
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{classroom_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    dummy_teacher_id = str(uuid.uuid4())
    try:
        supabase_new.table('subjects').upsert({
            "id": subject_id, 
            "subject_name": f"Subject {subject_id}",
            "teacher_id": dummy_teacher_id
        }).execute()
        
        # Upload to Supabase Storage
        bucket_name = "class_materials"
        try:
            # Create bucket if it doesn't exist
            supabase_new.storage.create_bucket(bucket_name, options={"public": True})
        except Exception:
            pass # Bucket likely already exists
            
        storage_path = f"{classroom_id}_{file.filename}"
        with open(file_path, "rb") as f:
            supabase_new.storage.from_(bucket_name).upload(storage_path, f.read(), {"content-type": "application/pdf"})
            
        pdf_url = supabase_new.storage.from_(bucket_name).get_public_url(storage_path)

        session_data = {
            "id": classroom_id,
            "subject_id": subject_id,
            "teacher_id": dummy_teacher_id,
            "topic_name": topic_title,
            "pdf_url": pdf_url
        }
        supabase_new.table('classrooms').insert(session_data).execute()

        
        from llama_parse import LlamaParse
        
        parser = LlamaParse(
            api_key=os.environ.get("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            verbose=True
        )
        
        llama_docs = parser.load_data(file_path)
        
        formatted_pages = []
        for i, doc in enumerate(llama_docs):
            page_num = i + 1
            formatted_pages.append(f"[--- PAGE {page_num} START ---]\n{doc.text}\n[--- PAGE {page_num} END ---]")
            
        full_text = "\n\n".join(formatted_pages)
                
        if not full_text.strip():
             raise ValueError("Could not extract any text or image data from the file.")
             
        # Save directly to memory (No FAISS)
        classroom_brains[classroom_id] = full_text
        
        # --- Full Text Cloud Backup ---
        bucket_vector = "vector_stores"
        try:
            supabase_new.storage.create_bucket(bucket_vector, options={"public": False})
        except Exception:
            pass # Bucket exists
            
        txt_path = f"{classroom_id}_fulltext.txt"
        supabase_new.storage.from_(bucket_vector).upload(txt_path, full_text.encode('utf-8'), {"content-type": "text/plain"})
        # --------------------------
            
        return {
            "status": "success", 
            "classroom_id": classroom_id,
            "message": f"Successfully uploaded {file.filename} and mapped systematically to Vector DB!"
        }
    except Exception as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"detail": str(e)})
