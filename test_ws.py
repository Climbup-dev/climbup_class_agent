import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://127.0.0.1:8000/ws/classroom/0f4ffbba-6df9-436a-b916-ba075af7e084?student_id=test&student_name=test"
    async with websockets.connect(uri) as websocket:
        msg1 = await websocket.recv()
        print("Init message:", msg1)
        
        await websocket.send("Give me Assignment question")
        
        msg2 = await websocket.recv()
        print("Response message:", msg2)

asyncio.get_event_loop().run_until_complete(test_ws())
