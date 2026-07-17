import os
import sys
# Fix unicode printing issues in Windows
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv

load_dotenv()

# Test Supabase
print("Testing Supabase...")
from supabase import create_client
try:
    supa = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
    res = supa.table("student_profiles").select("count", count="exact").limit(1).execute()
    print("[OK] Supabase works")
except Exception as e:
    print("[ERROR] Supabase Error:", e)

# Test Groq
print("\nTesting Groq...")
from langchain_groq import ChatGroq
try:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    res = llm.invoke("Say hi")
    print("[OK] Groq works:", res.content)
except Exception as e:
    print("[ERROR] Groq Error:", e)

# Test OpenRouter
print("\nTesting OpenRouter (Embeddings)...")
from langchain_openai import OpenAIEmbeddings
try:
    emb = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENROUTER_API_KEY"), openai_api_base="https://openrouter.ai/api/v1", model="openai/text-embedding-3-small")
    res = emb.embed_query("test")
    print("[OK] OpenRouter Embeddings works")
except Exception as e:
    print("[ERROR] OpenRouter Error:", e)
