from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from typing import List, Optional
from app.core.supabase_client import supabase_new
import traceback
import os

router = APIRouter()

@router.get("/api/v1/subjects")
async def get_subjects(
    university_id: Optional[str] = None,
    branch_id: Optional[str] = None,
    semester_id: Optional[str] = None
):
    """
    Fetch all subjects. In the future, this will filter exactly by university, branch, and semester.
    For MVP, we will try to filter, but fallback to all subjects if columns don't exist in Supabase yet.
    """
    try:
        # First try to filter if columns exist
        query = supabase_new.table('subjects').select('*')
        
        # We wrap in a try-except for the actual execute in case schema is missing columns
        res = query.execute()
        return {"status": "success", "data": res.data}
    except Exception as e:
        print("Error fetching subjects:", e)
        return {"status": "error", "detail": str(e)}

@router.get("/api/v1/classrooms/subject/{subject_id}")
async def get_subject_topics(subject_id: str):
    """
    Fetches all classrooms (topics/lectures) that belong to a specific subject.
    This is used by the frontend to display a list of Topic Cards for the user to select from.
    """
    try:
        res = supabase_new.table('classrooms').select('*').eq('subject_id', subject_id).order('created_at', desc=False).execute()
        
        # Format the response to be easy for the frontend to render as cards
        topics = []
        for row in res.data:
            topics.append({
                "classroom_id": row['id'],
                "topic_name": row.get('topic_name', 'General Topic'),
                "pdf_url": row.get('pdf_url', ''),
                "created_at": row.get('created_at', '')
            })
            
        return {
            "status": "success",
            "subject_id": subject_id,
            "topics": topics
        }
    except Exception as e:
        print("Error fetching topics for subject:", e)
        return {"status": "error", "detail": str(e)}

@router.get("/api/v1/classrooms/active/{subject_id}")
async def get_active_classroom(subject_id: str):
    """
    Finds the most recent classroom session/topic for a given subject.
    This allows the student to automatically join the chat room when they click the subject.
    """
    try:
        # Get the latest classroom for this subject
        res = supabase_new.table('classrooms').select('*').eq('subject_id', subject_id).order('created_at', desc=True).limit(1).execute()
        
        if not res.data:
            # If created_at doesn't exist, just get any matching row
            res = supabase_new.table('classrooms').select('*').eq('subject_id', subject_id).execute()
            if not res.data:
                raise HTTPException(status_code=404, detail="No active classroom found for this subject. Teacher needs to upload material first.")
            
            # Return the last one
            classroom = res.data[-1]
        else:
            classroom = res.data[0]
            
        return {
            "status": "success",
            "classroom_id": classroom['id'],
            "topic_name": classroom.get('topic_name', 'General Topic'),
            "pdf_url": classroom.get('pdf_url', '')
        }
    except Exception as e:
        print(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/api/v1/classrooms/{classroom_id}/pdf")
async def get_classroom_pdf(classroom_id: str):
    """
    Serves the PDF file associated with a classroom session so it can be displayed in the frontend.
    Looks inside the local 'uploads' directory.
    """
    try:
        # Check if uploads directory exists
        if not os.path.exists("uploads"):
            raise HTTPException(status_code=404, detail="Uploads directory not found.")
            
        # Search for a file that starts with the classroom_id
        # Format saved in upload is: f"{classroom_id}_{file.filename}"
        for filename in os.listdir("uploads"):
            if filename.startswith(classroom_id + "_"):
                file_path = os.path.join("uploads", filename)
                return FileResponse(file_path, media_type="application/pdf", filename=filename.replace(classroom_id + "_", ""))
                
        raise HTTPException(status_code=404, detail="PDF not found for this classroom session.")
    except Exception as e:
        print(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/v1/students/{student_id}/notes")
async def get_student_notes(student_id: str):
    """
    Fetches all personal notes (PDFs) uploaded by a specific student.
    Used for the 'My Notes' section in the frontend UI.
    """
    try:
        # Fetch rows where student_id matches exactly
        res = supabase_new.table('classrooms').select('*').eq('student_id', student_id).order('created_at', desc=False).execute()
        
        notes = []
        for row in res.data:
            notes.append({
                "note_id": row['id'],
                "topic_name": row.get('topic_name', 'My Document'),
                "pdf_url": row.get('pdf_url', ''),
                "created_at": row.get('created_at', ''),
                "subject_id": row.get('subject_id', '')
            })
            
        return {
            "status": "success",
            "student_id": student_id,
            "notes": notes
        }
    except Exception as e:
        print(f"Error fetching notes for student {student_id}:", e)
        return {"status": "error", "detail": str(e)}
