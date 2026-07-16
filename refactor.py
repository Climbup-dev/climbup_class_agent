import os
import glob

def replace_in_file(fpath):
    if not os.path.exists(fpath): return
    with open(fpath, 'r', encoding='utf-8') as f:
        c = f.read()
    c = c.replace('from langchain_openai import OpenAIEmbeddings
import os', 'from langchain_openai import OpenAIEmbeddings
import os')
    
    # Various forms of initialization
    c = c.replace('GoogleGenerativeAIEmbeddings(model="models/embedding-001", \ngoogle_api_key=os.getenv("GEMINI_API_KEY"))', 'OpenAIEmbeddings(openai_api_key=os.environ.get("OPENROUTER_API_KEY"), openai_api_base="https://openrouter.ai/api/v1", model="openai/text-embedding-3-small")')
    c = c.replace('OpenAIEmbeddings(openai_api_key=os.environ.get("OPENROUTER_API_KEY"), openai_api_base="https://openrouter.ai/api/v1", model="openai/text-embedding-3-small")', 'OpenAIEmbeddings(openai_api_key=os.environ.get("OPENROUTER_API_KEY"), openai_api_base="https://openrouter.ai/api/v1", model="openai/text-embedding-3-small")')
    c = c.replace('GoogleGenerativeAIEmbeddings(\n        model="models/embedding-001", \n        google_api_key=os.getenv("GEMINI_API_KEY")\n    )', 'OpenAIEmbeddings(openai_api_key=os.environ.get("OPENROUTER_API_KEY"), openai_api_base="https://openrouter.ai/api/v1", model="openai/text-embedding-3-small")')
    c = c.replace('OpenAIEmbeddings(openai_api_key=os.environ.get("OPENROUTER_API_KEY"), openai_api_base="https://openrouter.ai/api/v1", model="openai/text-embedding-3-small")', 'OpenAIEmbeddings(openai_api_key=os.environ.get("OPENROUTER_API_KEY"), openai_api_base="https://openrouter.ai/api/v1", model="openai/text-embedding-3-small")')
    
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(c)

for f in glob.glob('**/*.py', recursive=True):
    replace_in_file(f)
