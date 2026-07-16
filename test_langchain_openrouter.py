import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings

api_key = os.environ.get("OPENROUTER_API_KEY")

embeddings = OpenAIEmbeddings(
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    model="openai/text-embedding-3-small"
)

print("Testing Langchain OpenAI Embeddings via OpenRouter...")
try:
    # Need to pass http_client with verify=False if there are local SSL issues
    import httpx
    http_client = httpx.Client(verify=False)
    
    embeddings_with_client = OpenAIEmbeddings(
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model="openai/text-embedding-3-small",
        http_client=http_client
    )
    
    res = embeddings_with_client.embed_query("Hello world")
    print("SUCCESS! Dimension:", len(res))
except Exception as e:
    print("ERROR:", e)
