from fastapi import APIRouter
from app.core.memory import classroom_brains

router = APIRouter()

@router.get("/api/v1/debug/search/{classroom_id}")
def search_docstore(classroom_id: str):
    retriever = classroom_brains.get(classroom_id)
    if not retriever:
        return {"error": "Not found"}
    faiss_vs = retriever.retrievers[1].vectorstore
    docstore = faiss_vs.docstore._dict
    matches = []
    for doc_id, doc in docstore.items():
        if "assignment" in doc.page_content.lower() or "question" in doc.page_content.lower():
            matches.append(doc.page_content)
    return {"matches": matches}
