import os
import glob

def replace_in_file(fpath):
    if not os.path.exists(fpath): return
    with open(fpath, 'r', encoding='utf-8') as f:
        c = f.read()
    c = c.replace('from langchain_community.embeddings import HuggingFaceEmbeddings', 'from langchain_community.embeddings import HuggingFaceEmbeddings')
    
    # Various forms of initialization
    c = c.replace('GoogleGenerativeAIEmbeddings(model="models/embedding-001", \ngoogle_api_key=os.getenv("GEMINI_API_KEY"))', 'HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")')
    c = c.replace('HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")', 'HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")')
    c = c.replace('GoogleGenerativeAIEmbeddings(\n        model="models/embedding-001", \n        google_api_key=os.getenv("GEMINI_API_KEY")\n    )', 'HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")')
    c = c.replace('HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")', 'HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")')
    
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(c)

for f in glob.glob('**/*.py', recursive=True):
    replace_in_file(f)
