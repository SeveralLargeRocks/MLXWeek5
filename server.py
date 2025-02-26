from fastapi import FastAPI, WebSocket
import diarize
import io

import os
dirname = os.path.dirname(__file__)

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        bytes = await websocket.receive_bytes()
        # audio_stream = io.BytesIO(bytes)
        # print(type(audio_stream))
        diarize.handle_socket_message(bytes)
        await websocket.send_text(f"Message text was: {bytes}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
