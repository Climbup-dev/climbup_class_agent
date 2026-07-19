"""
Smart Tour Generator - Downloads FAISS from Supabase for scanned PDFs.
Uses OpenRouter (separate rate limit from Groq) as LLM.
Run: python generate_tours.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from app.core.ssl_fix import apply_ssl_fix
apply_ssl_fix()

import json
import logging
import zipfile
import shutil
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Use Gemini 3.1 Flash Lite - highly reliable and fast
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)
llm = llm.bind(response_format={"type": "json_object"})

CLASSROOMS_TO_PROCESS = [
    "57925be6-10f3-4d62-9acc-a4c8657006a8",  # Cyber Jurisdiction (scanned)
    "43e76c09-cc50-4a26-aba5-b7a09c59de67",  # DCN unit 2
    "d13e5621-be69-4bb6-b7f1-00a7ba55c824",  # Cyber Jurisdiction copy
    "6dd87c6b-4a0a-4919-af60-59a37b59b314",  # DCN copy
    "0f4ffbba-6df9-436a-b916-ba075af7e084",
]

TOUR_PROMPT = PromptTemplate.from_template("""
You are a super cool, ultra-smart Indian professor. You have been given the full text of a lecture PDF.
Your task is to convert this ENTIRE PDF into a highly detailed, structured "Guided Tour" course.

PDF TEXT:
{full_text}

IMPORTANT RULES:
1. DO NOT MISS A SINGLE PARAGRAPH: You MUST generate a separate lesson for EVERY SINGLE paragraph, heading, and concept in the PDF sequentially. If the PDF has 50 paragraphs, there MUST be 50 lessons. Leave nothing behind!
2. EXACT QUOTE: For "exact_quote", copy and paste the ENTIRE relevant paragraph verbatim from the text. Do NOT summarize or change a single word.
3. HINGLISH EXPLANATION: The "explanation" MUST be in a flawless, natural mix of Hindi and English (Hinglish). 
4. DETAILED & FUN: The explanation should be in FULL DETAIL. Break down the concept using funny, real-world, everyday Indian analogies so the student feels "WOW, this is so easy to understand!". Use emojis to make it expressive.

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


def get_text_from_pdf(classroom_id, uploads_dir):
    """Try PyMuPDF first (text PDFs), else download FAISS from Supabase (scanned PDFs)."""
    # Strategy 1: PyMuPDF
    for fname in os.listdir(uploads_dir):
        if fname.startswith(classroom_id) and fname.endswith(".pdf"):
            pdf_path = os.path.join(uploads_dir, fname)
            doc = fitz.open(pdf_path)
            text = "".join([page.get_text() for page in doc])
            if text.strip():
                logging.info(f"✅ PyMuPDF extracted {len(text)} chars")
                return text
            logging.warning("⚠️  PyMuPDF got no text (scanned PDF). Trying FAISS...")
            break

    # Strategy 2: Download FAISS from Supabase and reconstruct text from chunks
    logging.info(f"📥 Downloading FAISS from Supabase for {classroom_id}...")
    try:
        from app.core.supabase_client import supabase_new
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings

        zip_name = f"faiss_{classroom_id}.zip"
        faiss_dir = f"faiss_{classroom_id}"

        # Download zip
        data = supabase_new.storage.from_("vector_stores").download(zip_name)
        with open(zip_name, "wb") as f:
            f.write(data)

        # Extract
        with zipfile.ZipFile(zip_name, "r") as z:
            z.extractall(faiss_dir)
        os.remove(zip_name)

        # Load FAISS and get all chunk texts
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            model="openai/text-embedding-3-small"
        )
        vs = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
        docs = vs.docstore._dict.values()
        full_text = "\n\n".join([doc.page_content for doc in docs])

        shutil.rmtree(faiss_dir, ignore_errors=True)
        logging.info(f"✅ Reconstructed {len(full_text)} chars from FAISS chunks")
        return full_text

    except Exception as e:
        logging.error(f"❌ FAISS download failed: {e}")
        return ""


uploads_dir = "uploads"

for classroom_id in CLASSROOMS_TO_PROCESS:
    tour_path = os.path.join(uploads_dir, f"{classroom_id}_tour.json")

    if os.path.exists(tour_path):
        logging.info(f"⏭️  Skipping {classroom_id} - tour already exists")
        continue

    logging.info(f"\n{'='*60}")
    logging.info(f"Processing: {classroom_id}")

    full_text = get_text_from_pdf(classroom_id, uploads_dir)

    if not full_text.strip():
        logging.error(f"❌ No text found for {classroom_id}. Skipping.")
        continue

    logging.info(f"🤖 Generating tour via OpenRouter...")
    try:
        response = llm.invoke(TOUR_PROMPT.format(full_text=full_text[:30000]))
        content = response.content
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
            
        with open(tour_path, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info(f"✅ Tour saved: {tour_path}")
    except Exception as e:
        logging.error(f"❌ LLM failed: {e}")

logging.info("\n🎉 All done!")
