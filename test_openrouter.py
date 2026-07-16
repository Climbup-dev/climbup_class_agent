import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")

if not api_key:
    print("NO OPENROUTER API KEY FOUND")
    exit(1)

url = "https://openrouter.ai/api/v1/embeddings"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "openai/text-embedding-3-small",
    "input": "Hello world"
}

print("Testing OpenRouter Embeddings...")
res = requests.post(url, headers=headers, json=data, verify=False)
print("STATUS:", res.status_code)
try:
    j = res.json()
    if 'data' in j:
        print("SUCCESS! Dimension:", len(j['data'][0]['embedding']))
    else:
        print("RESPONSE:", json.dumps(j, indent=2))
except Exception as e:
    print("ERROR:", e, res.text)
