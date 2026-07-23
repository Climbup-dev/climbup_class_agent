import requests

url = "https://climbup-class-agent.onrender.com/api/v1/upload-smart"

response = requests.post(url, data={
    "university_id": "test",
    "branch_id": "test",
    "semester_id": "test",
    "subject_id": "test",
    "topic_title": "test"
})

print(f"Status: {response.status_code}")
print(f"Text: {response.text}")
