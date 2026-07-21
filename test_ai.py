import asyncio
import os
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

from app.ai.graph import classroom_app

async def main():
    print("Testing Classroom AI Agent...")
    
    # Create a mock state
    dummy_context = """
    [Source: Page 1]
    TCP (Transmission Control Protocol) is a standard that defines how to establish and maintain a network conversation through which application programs can exchange data. TCP works with the Internet Protocol (IP), which defines how computers send packets of data to each other.
    
    [Source: Page 2]
    UDP (User Datagram Protocol) is an alternative communications protocol to Transmission Control Protocol (TCP) used primarily for establishing low-latency and loss-tolerating connections between applications on the internet.
    """
    
    state = {
        "classroom_id": "test_class_123",
        "subject_name": "Computer Networks",
        "topic_name": "Transport Layer Protocols",
        "lecture_date": "Today",
        "active_students": "John Doe",
        "student_name": "John Doe",
        "student_profile": "Engagement Level: High",
        "chat_history": "No previous history.",
        "question": "What is the main difference between TCP and UDP based on the notes?",
        "context": dummy_context,
        "used_analogies": [],
        "strike_count": 0,
        "is_disruptive": False,
        "is_abusive": False
    }
    
    try:
        print("\nInvoking AI with dummy context...")
        result = await classroom_app.ainvoke(state)
        
        print("\n--- RESULTS ---")
        print("Chat Content:")
        print(result.get("chat_content", "No chat content returned"))
        
        print("\nBoard Content:")
        print(result.get("board_content", "No board content returned"))
        
        print("\nSUCCESS! The AI pipeline is working perfectly.")
    except Exception as e:
        print(f"\n❌ ERROR during invocation: {e}")

if __name__ == "__main__":
    asyncio.run(main())
