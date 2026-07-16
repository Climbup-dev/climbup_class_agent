from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from app.core.supabase_client import supabase_new
import traceback

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
            "topic_name": classroom.get('topic_name', 'General Topic')
        }
    except Exception as e:
        print(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=str(e))
