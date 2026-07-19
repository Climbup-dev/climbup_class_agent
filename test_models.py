import os, requests
key = os.environ.get("GOOGLE_API_KEY")
res = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}")
data = res.json()
print([m["name"] for m in data.get("models", [])])
