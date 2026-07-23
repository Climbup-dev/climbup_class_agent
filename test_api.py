import requests
import json

url = "https://climbup-class-agent.onrender.com/api/v1/students/b3e09339-ff6b-4e88-9d8a-6ddfcb8eddb1/notes"
res = requests.get(url)
print(f"Status: {res.status_code}")
try:
    print(json.dumps(res.json(), indent=2))
except:
    print(res.text)
