import requests
import json
import os

url = "http://127.0.0.1:8000/api/v1/upload-smart"
file_path = "c:/Users/shaik/Desktop/mini project report.pdf"

if not os.path.exists(file_path):
    print("File not found")
    exit(1)

print(f"Uploading {file_path} to {url}...")
try:
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {
            "university_id": "dummy_uni",
            "branch_id": "dummy_branch",
            "semester_id": "dummy_sem",
            "subject_id": "a68ca16c-d66c-4f79-b594-e8698b5cae0e",
            "topic_title": "Test Topic"
        }
        res = requests.post(url, files=files, data=data)
        
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.text}")
    
    if res.status_code == 200:
        resp_data = res.json()
        classroom_id = resp_data.get("classroom_id")
        print(f"Classroom ID: {classroom_id}")
        
        if classroom_id:
            json_path = f"c:/Users/shaik/Desktop/class agent/backend/uploads/{classroom_id}_tour.json"
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as jf:
                    tour_data = json.load(jf)
                    print(f"Total Lessons: {len(tour_data.get('lessons', []))}")
                    found_diagram = False
                    for i, lesson in enumerate(tour_data.get("lessons", [])):
                        if "![Diagram]" in lesson.get("exact_quote", ""):
                            print(f"Lesson {i+1} has a Diagram!")
                            found_diagram = True
                    if not found_diagram:
                        print("NO DIAGRAMS FOUND IN JSON.")
            else:
                print(f"JSON not found at {json_path}")
                
except Exception as e:
    print(f"Error: {e}")
