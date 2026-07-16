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
        
        full_text = "\n\n".join([doc.text for doc in llama_docs])
                
        if not full_text.strip():
             raise ValueError("Could not extract any text or image data from the PDF.")
             
        document = Document(page_content=full_text, metadata={"classroom_id": classroom_id, "source": file.filename})
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents([document])
        
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENROUTER_API_KEY"), openai_api_base="https://openrouter.ai/api/v1", model="openai/text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embeddings)
        classroom_brains[classroom_id] = vector_store
            
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
        full_text = "\n\n".join([doc.text for doc in llama_docs])
                
        if not full_text.strip():
             raise ValueError("Could not extract any text or image data from the file.")
             
        metadata = {
            "classroom_id": classroom_id, 
            "university_id": university_id,
            "branch_id": branch_id,
            "semester_id": semester_id,
            "subject_id": subject_id,
            "topic_title": topic_title,
            "source": file.filename
        }
        document = Document(page_content=full_text, metadata=metadata)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents([document])
        
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENROUTER_API_KEY"), openai_api_base="https://openrouter.ai/api/v1", model="openai/text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embeddings)
        classroom_brains[classroom_id] = vector_store
        
        # --- FAISS Cloud Backup ---
        faiss_dir = f"faiss_{classroom_id}"
        vector_store.save_local(faiss_dir)
        
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
        # --------------------------
            
        return {
            "status": "success", 
            "classroom_id": classroom_id,
            "message": f"Successfully uploaded {file.filename} and mapped systematically to Vector DB!"
        }
    except Exception as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"detail": str(e)})
