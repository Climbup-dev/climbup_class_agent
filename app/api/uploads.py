import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from supabase import create_client, Client
from app.core.config import settings
from app.core.database import get_db
from app.models.classroom import Classroom, Material
from app.rag.loader import extract_documents
from app.rag.embedder import process_and_store_documents

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/classrooms/{classroom_id}/upload")
async def upload_material(classroom_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 1. Verify classroom exists
    classroom = db.query(Classroom).filter(Classroom.id == classroom_id).first()
    if not classroom:
        raise HTTPException(status_code=404, detail="Classroom not found")
        
    # 2. Save file locally
    file_ext = file.filename.split(".")[-1].lower()
    file_path = os.path.join(UPLOAD_DIR, f"{classroom_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 3. Extract text
        metadata = {"source": file.filename, "classroom_id": classroom_id}
        documents = extract_documents(file_path, file_ext, metadata)
        
        # 4. Upload file to Supabase Storage
        supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        storage_path = f"classrooms/{classroom_id}/{file.filename}"
        
        with open(file_path, "rb") as f:
            # We assume a bucket named 'materials' exists
            supabase.storage.from_("materials").upload(storage_path, f.read())
            
        public_url = supabase.storage.from_("materials").get_public_url(storage_path)
        
        # 5. Create material record
        material = Material(
            classroom_id=classroom_id,
            file_url=public_url,
            file_type=file_ext,
            processing_status="PROCESSED"
        )
        db.add(material)
        db.commit()
        db.refresh(material)
        
        # Update metadata with material id and process
        for doc in documents:
            doc.metadata["material_id"] = material.id
            
        # 6. Process and store in Vector DB
        process_and_store_documents(documents, classroom_id)
        
        # Clean up local file
        os.remove(file_path)
        
        return {"message": "File uploaded to Supabase and processed successfully", "material_id": material.id}
        
    except Exception as e:
        # If material was created, mark as error
        material = db.query(Material).filter(Material.classroom_id == classroom_id).order_by(Material.id.desc()).first()
        if material and material.processing_status != "PROCESSED":
            material.processing_status = "ERROR"
            db.commit()
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
