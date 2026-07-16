import asyncio
from app.api.classrooms import get_subjects, get_active_classroom

async def test():
    print("Testing get_subjects()...")
    subjects = await get_subjects()
    print("Subjects returned:", subjects)
    
    if subjects.get('data') and len(subjects['data']) > 0:
        first_sub = subjects['data'][0]['id']
        print(f"\nTesting get_active_classroom for subject: {first_sub}")
        
        try:
            room = await get_active_classroom(first_sub)
            print("Active classroom:", room)
        except Exception as e:
            print("Error finding active classroom:", e)

if __name__ == "__main__":
    asyncio.run(test())
