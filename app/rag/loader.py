import os
import fitz  # PyMuPDF
from pptx import Presentation
from langchain.schema import Document

def parse_pdf(file_path: str) -> str:
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def parse_ppt(file_path: str) -> str:
    text = ""
    try:
        prs = Presentation(file_path)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
    except Exception as e:
        print(f"Error reading PPT: {e}")
    return text

def extract_documents(file_path: str, file_type: str, metadata: dict) -> list[Document]:
    text = ""
    if file_type.lower() == 'pdf':
        text = parse_pdf(file_path)
    elif file_type.lower() in ['ppt', 'pptx']:
        text = parse_ppt(file_path)
    else:
        # Fallback to reading as text
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            
    if not text.strip():
        return []
        
    return [Document(page_content=text, metadata=metadata)]
