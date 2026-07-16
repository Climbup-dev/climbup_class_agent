import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from langchain_community.vectorstores import FAISS
from app.core.ssl_fix import apply_ssl_fix
from dotenv import load_dotenv

apply_ssl_fix()
load_dotenv()

try:
    file_path = "uploads/123_Cyber Jurisdiction .pdf"
    
    print("Loading PDF...")
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    
    print("Splitting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    
    print("Embedding and storing in FAISS...")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENROUTER_API_KEY"), openai_api_base="https://openrouter.ai/api/v1", model="openai/text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
