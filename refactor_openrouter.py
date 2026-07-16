import os
import glob

def replace_in_file(fpath):
    if not os.path.exists(fpath): return
    with open(fpath, 'r', encoding='utf-8') as f:
        c = f.read()
        
    if 'HuggingFaceEmbeddings' not in c:
        return
        
    c = c.replace('from langchain_openai import OpenAIEmbeddings
import os', 'from langchain_openai import OpenAIEmbeddings\nimport os')
    
    # Replace the instantiation
    c = c.replace('OpenAIEmbeddings(openai_api_key=os.environ.get("OPENROUTER_API_KEY"), openai_api_base="https://openrouter.ai/api/v1", model="openai/text-embedding-3-small")', 'OpenAIEmbeddings(openai_api_key=os.environ.get("OPENROUTER_API_KEY"), openai_api_base="https://openrouter.ai/api/v1", model="openai/text-embedding-3-small")')
    
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(c)

for f in glob.glob('**/*.py', recursive=True):
    replace_in_file(f)
