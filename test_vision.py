import os
import base64
from app.core.llm_balancer import get_balanced_vision_llm
from langchain_core.messages import HumanMessage

def test_vision():
    try:
        llm = get_balanced_vision_llm()
        print(f"Loaded Vision Model: {llm.model}")
        
        # Create a tiny 1x1 pixel PNG in base64
        tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        
        response = llm.invoke([
            HumanMessage(content=[
                {"type": "text", "text": "What is this image? Reply in one word."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tiny_png}"}}
            ])
        ])
        print("Response:", response.content)
    except Exception as e:
        print("ERROR:", str(e))

if __name__ == "__main__":
    test_vision()
