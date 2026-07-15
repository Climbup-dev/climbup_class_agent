from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import fitz
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.core.memory import classroom_brains
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"), transport="rest")

router = APIRouter()

@router.post("/api/v1/classrooms/{classroom_id}/upload")
async def upload_material(classroom_id: str, file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported for the MVP.")
        
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
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        classroom_brains[classroom_id] = vector_store
            
        return {"status": "success", "message": f"Successfully uploaded and trained AI on {file.filename} using Gemini OCR!"}
    except Exception as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"detail": str(e)})
