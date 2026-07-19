import os
import sys
from app.core.ssl_fix import apply_ssl_fix
apply_ssl_fix()
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

def inspect_pdf_llamaparse(file_path: str):
    print(f"--- INSPECTING PDF WITH LLAMAPARSE (Exactly how API works): {file_path} ---")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print("\n[Step 1] Calling LlamaParse API... (This might take 10-30 seconds)")
    try:
        parser = LlamaParse(
            api_key=os.environ.get("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            verbose=True
        )
        llama_docs = parser.load_data(file_path)
    except Exception as e:
        print("LlamaParse API Error:", e)
        return

    full_text = "\n\n".join([doc.text for doc in llama_docs])
    
    if not full_text.strip():
        print("Could not extract any text or image data from the file.")
        return

    print(f"Extraction Successful! Total extracted characters: {len(full_text)}")
    print("\n--- RAW TEXT PREVIEW (First 500 chars) ---")
    print(full_text[:500] + "...\n[TRUNCATED]")
    
    # 2. Test Chunking
    print("\n[Step 2] Chunking text...")
    document = Document(page_content=full_text, metadata={"source": file_path})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents([document])
    
    print(f"Total Chunks Created: {len(chunks)}")
    
    print(f"\n--- SHOWING ALL {len(chunks)} CHUNKS EXACTLY AS THEY GO TO VECTOR DB ---")
    for i in range(len(chunks)):
        print(f"\n================ CHUNK {i+1} ================")
        print(chunks[i].page_content)
        print(f"--- Metadata: {chunks[i].metadata} ---")
        print("==============================================")

if __name__ == "__main__":
    test_file = "uploads/2ca7da32-afba-4282-ab63-1e66d30466f7_Cyber Jurisdiction .pdf"
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    inspect_pdf_llamaparse(test_file)
