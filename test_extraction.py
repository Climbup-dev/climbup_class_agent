import os
import asyncio
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv()

# We will just test the PyMuPDF image extraction and vision model invocation
import fitz
import base64
from langchain_core.messages import HumanMessage
from app.core.llm_balancer import get_balanced_vision_llm

def test_extraction():
    file_path = "c:/Users/shaik/Desktop/mini project report.pdf"
    if not os.path.exists(file_path):
        print("File not found")
        return

    try:
        pdf_document = fitz.open(file_path)
        print(f"Total pages: {len(pdf_document)}")
        
        all_extracted_images = []
        for page_num in range(min(2, len(pdf_document))): # Only check first 2 pages
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)
            print(f"Page {page_num+1} has {len(image_list)} images.")
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                print(f"Extracted image {img_index} with ext {image_ext}")
                
                # Test Vision LLM
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                llm_vision = get_balanced_vision_llm()
                print(f"Invoking {llm_vision.model}...")
                response = llm_vision.invoke([
                    HumanMessage(content=[
                        {"type": "text", "text": "Describe this image in detail. Reply IGNORE if it's just text."},
                        {"type": "image_url", "image_url": {"url": f"data:image/{image_ext};base64,{base64_image}"}}
                    ])
                ])
                print(f"Description: {response.content.strip()[:100]}...")
                
    except Exception as e:
        print(f"Error during extraction: {e}")

if __name__ == "__main__":
    test_extraction()
