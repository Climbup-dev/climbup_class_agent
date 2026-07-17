import urllib.parse
import requests

def test_pollinations():
    keyword = "cyber attack ransomware digital art"
    encoded = urllib.parse.quote(keyword)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=800&height=400&nologo=true"
    
    # Just check if it returns 200
    res = requests.get(url)
    print("Status:", res.status_code)
    print("Content Type:", res.headers.get("content-type"))
    print("URL:", url)

test_pollinations()
