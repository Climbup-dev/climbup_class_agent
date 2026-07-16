import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("NO GEMINI API KEY FOUND IN .ENV")
    exit(1)

def test_model(model_name):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:embedContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": f"models/{model_name}",
        "content": {
            "parts": [{"text": "Hello world"}]
        }
    }
    print(f"Testing {model_name}...")
    res = requests.post(url, headers=headers, json=data, verify=False)
    print("STATUS:", res.status_code)
    try:
        print("RESPONSE:", json.dumps(res.json(), indent=2))
    except:
        print("RESPONSE:", res.text)
    print("-" * 50)

test_model("embedding-001")
test_model("text-embedding-004")
