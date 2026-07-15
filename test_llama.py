import os
from dotenv import load_dotenv
from app.core.ssl_fix import apply_ssl_fix
apply_ssl_fix()

from llama_parse import LlamaParse

load_dotenv()
api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
print(f"API Key present: {api_key is not None}")

try:
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        verbose=True
    )
    
    # Test with a small PDF
    file_path = r"C:\Users\shaik\Desktop\class agent\backend\uploads\123_Define_ i) Magnetic permeability. ii) Hysterics iii) Retentivity iv) Coercivity.pdf"
    print(f"Parsing file: {file_path}")
    
    docs = parser.load_data(file_path)
    
    full_text = "\n\n".join([doc.text for doc in docs])
    print("\n--- EXTRACTION SUCCESSFUL ---")
    print(f"Extracted {len(full_text)} characters.")
    print("Preview:\n" + full_text[:500])
except Exception as e:
    print(f"ERROR: {e}")
