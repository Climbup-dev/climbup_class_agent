import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from app.ai.graph import persona_node

state = {
    "teaching_strategy": "Test strategy",
    "student_name": "Test Name",
    "student_profile": "Test Profile",
    "awarded_xp": 10,
    "image_url": "http://test.com"
}
try:
    res = persona_node(state)
    print("Result:", res)
except Exception as e:
    print("Exception during persona_node:", e)
