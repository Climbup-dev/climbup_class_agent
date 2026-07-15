import os
from dotenv import load_dotenv
from app.core.ssl_fix import apply_ssl_fix
apply_ssl_fix()

from llama_parse import LlamaParse

load_dotenv()
api_key = os.environ.get("LLAMA_CLOUD_API_KEY")

try:
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        verbose=True
    )
    
    file_path = r"C:\Users\shaik\Desktop\class agent\backend\uploads\123_Cyber Jurisdiction .pdf"
    print(f"Parsing file: {file_path}")
    
    docs = parser.load_data(file_path)
    print("EXTRACTION SUCCESSFUL")
except Exception as e:
    print(f"ERROR: {e}")
