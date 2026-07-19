import os
import random
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import logging

load_dotenv()

def get_api_keys(env_var_name: str) -> list[str]:
    keys_str = os.environ.get(env_var_name, "")
    keys_str = keys_str.replace('"', '').replace("'", "").strip()
    if not keys_str:
        return []
    return [k.strip() for k in keys_str.split(",") if k.strip()]

def create_universal_fallback_chain(temperature=0.7, bind_kwargs=None):
    llms = []
    
    # 1. SambaNova
    sambanova_keys = get_api_keys("SAMBANOVA_API_KEYS")
    if sambanova_keys:
        random.shuffle(sambanova_keys)
        for key in sambanova_keys:
            llm = ChatOpenAI(
                model="Meta-Llama-3.3-70B-Instruct", 
                openai_api_key=key, 
                openai_api_base="https://api.sambanova.ai/v1",
                temperature=temperature,
                max_retries=0 # Instant failover
            )
            if bind_kwargs: llm = llm.bind(**bind_kwargs)
            llms.append(llm)

    # 2. Cerebras
    cerebras_keys = get_api_keys("CEREBRAS_API_KEYS")
    if cerebras_keys:
        random.shuffle(cerebras_keys)
        for key in cerebras_keys:
            llm = ChatOpenAI(
                model="llama3.3-70b", 
                openai_api_key=key, 
                openai_api_base="https://api.cerebras.ai/v1",
                temperature=temperature,
                max_retries=0
            )
            if bind_kwargs: llm = llm.bind(**bind_kwargs)
            llms.append(llm)

    # 3. Groq
    groq_keys = get_api_keys("GROQ_API_KEYS")
    if groq_keys:
        random.shuffle(groq_keys)
        for key in groq_keys:
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=temperature,
                api_key=key,
                max_retries=0
            )
            if bind_kwargs: llm = llm.bind(**bind_kwargs)
            llms.append(llm)

    # 4. Google Gemini (Upgraded to 2.0 Flash)
    gemini_keys = get_api_keys("GEMINI_API_KEYS")
    if gemini_keys:
        random.shuffle(gemini_keys)
        for key in gemini_keys:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=temperature,
                api_key=key,
                max_retries=0
            )
            if bind_kwargs: llm = llm.bind(**bind_kwargs)
            llms.append(llm)
            
    # 5. Together AI
    together_keys = get_api_keys("TOGETHER_API_KEYS")
    if together_keys:
        random.shuffle(together_keys)
        for key in together_keys:
            llm = ChatOpenAI(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo", 
                openai_api_key=key, 
                openai_api_base="https://api.together.xyz/v1",
                temperature=temperature,
                max_retries=0
            )
            if bind_kwargs: llm = llm.bind(**bind_kwargs)
            llms.append(llm)
            
    # 6. Mistral AI
    mistral_keys = get_api_keys("MISTRAL_API_KEYS")
    if mistral_keys:
        random.shuffle(mistral_keys)
        for key in mistral_keys:
            llm = ChatOpenAI(
                model="mistral-small-latest", 
                openai_api_key=key, 
                openai_api_base="https://api.mistral.ai/v1",
                temperature=temperature,
                max_retries=0
            )
            if bind_kwargs: llm = llm.bind(**bind_kwargs)
            llms.append(llm)

    # 7. OpenRouter Free Tier Fallbacks
    openrouter_keys = get_api_keys("OPENROUTER_API_KEYS")
    if openrouter_keys:
        random.shuffle(openrouter_keys)
        for key in openrouter_keys:
            llm = ChatOpenAI(
                model="google/gemini-2.0-flash-lite-preview-02-05:free", 
                openai_api_key=key, 
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=temperature,
                max_retries=0
            )
            if bind_kwargs: llm = llm.bind(**bind_kwargs)
            llms.append(llm)

    if not llms:
        logging.warning("No API keys found in .env! Returning a dummy LLM.")
        return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key="sk-dummy")
        
    logging.info(f"Loaded {len(llms)} API keys into Universal Fallback Cluster.")
    
    if len(llms) == 1:
        return llms[0]
        
    return llms[0].with_fallbacks(llms[1:])

def get_balanced_vision_llm():
    gemini_keys = get_api_keys("GEMINI_API_KEYS")
    openrouter_keys = get_api_keys("OPENROUTER_API_KEYS")
    
    llms = []
    
    # Primary: Native Google Gemini 2.0 Flash
    if gemini_keys:
        random.shuffle(gemini_keys)
        for key in gemini_keys:
            llms.append(ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                temperature=0.3, 
                api_key=key, 
                max_retries=0 # Crucial: Instant failover to the next key!
            ))
            
    # Secondary: OpenRouter Fallbacks
    if openrouter_keys:
        random.shuffle(openrouter_keys)
        for key in openrouter_keys:
            # Fallback 1: OpenRouter Gemini 2.0
            llms.append(ChatOpenAI(
                model="google/gemini-2.0-flash-lite-preview-02-05:free", 
                openai_api_key=key, 
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.3,
                max_retries=0
            ))
            # Fallback 2: Qwen 2 VL (Reliable for vision)
            llms.append(ChatOpenAI(
                model="qwen/qwen-2-vl-7b-instruct:free", 
                openai_api_key=key, 
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.3,
                max_retries=0
            ))
            
    if not llms:
        logging.warning("No GEMINI or OPENROUTER keys found for Vision processing.")
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key="dummy")
        
    # Return the fallback chain
    return llms[0].with_fallbacks(llms[1:]) if len(llms) > 1 else llms[0]

def get_balanced_text_llm(model_name="N/A", temperature=0.7):
    return create_universal_fallback_chain(temperature=temperature)

def get_balanced_fast_llm():
    return create_universal_fallback_chain(
        temperature=0.7, 
        bind_kwargs={"response_format": {"type": "json_object"}}
    )
