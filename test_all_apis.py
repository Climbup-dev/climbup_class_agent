import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

def get_api_keys(env_var_name: str) -> list[str]:
    keys_str = os.environ.get(env_var_name, "")
    keys_str = keys_str.replace('"', '').replace("'", "").strip()
    if not keys_str:
        return []
    return [k.strip() for k in keys_str.split(",") if k.strip()]

async def test_llm(name, llm):
    try:
        print(f"Testing {name}...")
        res = await llm.ainvoke("Reply with the word 'SUCCESS' and nothing else.")
        if "SUCCESS" in res.content.upper():
            print(f"✅ {name}: WORKING")
        else:
            print(f"⚠️ {name}: RESPONDED BUT UNEXPECTED OUTPUT: {res.content}")
    except Exception as e:
        print(f"❌ {name}: FAILED - {str(e)}")

async def main():
    tasks = []
    
    # 1. Groq
    from langchain_groq import ChatGroq
    for i, key in enumerate(get_api_keys("GROQ_API_KEYS")):
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=key, max_retries=0)
        tasks.append(test_llm(f"Groq Key {i+1}", llm))
        
    # 2. Gemini
    from langchain_google_genai import ChatGoogleGenerativeAI
    for i, key in enumerate(get_api_keys("GEMINI_API_KEYS")):
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=key, max_retries=0)
        tasks.append(test_llm(f"Gemini Key {i+1}", llm))
        
    # 3. OpenRouter
    from langchain_openai import ChatOpenAI
    for i, key in enumerate(get_api_keys("OPENROUTER_API_KEYS")):
        llm = ChatOpenAI(model="google/gemini-2.0-flash:free", openai_api_key=key, openai_api_base="https://openrouter.ai/api/v1", max_retries=0)
        tasks.append(test_llm(f"OpenRouter Key {i+1}", llm))
        
    # 4. Cerebras
    for i, key in enumerate(get_api_keys("CEREBRAS_API_KEYS")):
        llm = ChatOpenAI(model="llama3.3-70b", openai_api_key=key, openai_api_base="https://api.cerebras.ai/v1", max_retries=0)
        tasks.append(test_llm(f"Cerebras Key {i+1}", llm))
        
    # 5. SambaNova
    for i, key in enumerate(get_api_keys("SAMBANOVA_API_KEYS")):
        llm = ChatOpenAI(model="Meta-Llama-3.3-70B-Instruct", openai_api_key=key, openai_api_base="https://api.sambanova.ai/v1", max_retries=0)
        tasks.append(test_llm(f"SambaNova Key {i+1}", llm))
        
    # 6. Together AI
    for i, key in enumerate(get_api_keys("TOGETHER_API_KEYS")):
        llm = ChatOpenAI(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", openai_api_key=key, openai_api_base="https://api.together.xyz/v1", max_retries=0)
        tasks.append(test_llm(f"Together Key {i+1}", llm))
        
    # 7. Mistral
    for i, key in enumerate(get_api_keys("MISTRAL_API_KEYS")):
        llm = ChatOpenAI(model="mistral-small-latest", openai_api_key=key, openai_api_base="https://api.mistral.ai/v1", max_retries=0)
        tasks.append(test_llm(f"Mistral Key {i+1}", llm))

    if tasks:
        await asyncio.gather(*tasks)
    else:
        print("No API keys found to test.")

if __name__ == "__main__":
    asyncio.run(main())
