from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
import os
print("Loading model...")
try:
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENROUTER_API_KEY"), openai_api_base="https://openrouter.ai/api/v1", model="openai/text-embedding-3-small")
    res = embeddings.embed_query("Hello world")
    print("SUCCESS! Embedded dimension:", len(res))
except Exception as e:
    print("ERROR:", e)
