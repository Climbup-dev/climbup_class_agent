import asyncio
from fastapi.testclient import TestClient
from app.main import app
import sys

client = TestClient(app)

def test_upload_missing_auth():
    print("Testing upload without auth...")
    response = client.post("/api/v1/upload-smart", data={
        "university_id": "u1",
        "branch_id": "b1",
        "semester_id": "s1",
        "subject_id": "sub1",
        "topic_title": "Test Topic"
    })
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

def test_upload_with_auth():
    print("\nTesting upload with auth...")
    response = client.post("/api/v1/upload-smart", 
        data={
            "university_id": "u1",
            "branch_id": "b1",
            "semester_id": "s1",
            "subject_id": "sub1",
            "topic_title": "Test Topic"
        },
        headers={
            "Authorization": "Bearer fake_token"
        },
        files={
            "file": ("test.pdf", b"dummy pdf content", "application/pdf")
        }
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

if __name__ == "__main__":
    test_upload_missing_auth()
    test_upload_with_auth()
