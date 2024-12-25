# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 11:20:46 2024

@author: mashn
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 15:11:19 2024

@author: mashn
"""

from threading import Thread
from typing import Union
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from model import load_quantized_llm_model, rag_pipeline



app = FastAPI()

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""

model_and_tokenizer = {"model": None, "tokenizer": None, "loaded": False}

rag_inference_pipeline = None




def start_llm_loading():
    global model_and_tokenizer
    try:
        model, tokenizer = load_quantized_llm_model()
        model_and_tokenizer["model"] = model
        model_and_tokenizer["tokenizer"] = tokenizer
        model_and_tokenizer["loaded"] = True
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
    
def is_model_ready():
    return model_and_tokenizer["loaded"]

@app.on_event("startup")
async def startup_event():
    global rag_inference_pipeline
    thread = Thread(target=start_llm_loading, daemon=True)
    thread.start()
    print("Waiting for model to load...")
    while not is_model_ready():
        pass
    
    try:
        # Initialize the RAG pipeline
        rag_inference_pipeline = rag_pipeline(model_and_tokenizer["model"], model_and_tokenizer["tokenizer"])
        print("RAG pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading RAG pipeline: {e}")
        rag_inference_pipeline = None



@app.get("/")
async def get():
    return HTMLResponse(html)



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    
    global rag_inference_pipeline
    await websocket.accept()
    
    
    if rag_inference_pipeline is None:
        await websocket.send_text("Error: RAG pipeline is not available.")
        await websocket.close()
        return
        
    try:
        while True:
            qs_data = await websocket.receive_text()
            
            # Ensure `invoke` is the correct method and accept parameters like `text`
            inference_result = rag_inference_pipeline.invoke(qs_data)
            
            if isinstance(inference_result, dict) and "result" in inference_result:
                result_text = inference_result["result"]
                
                if "Helpful Answer:" in result_text:
                    parts = result_text.split("Helpful Answer:", 1)
                    context = parts[0].strip()
                    helpful_answer = parts[1].strip()
                else:
                    context = result_text.strip()
                    helpful_answer = "No helpful answer available."
            else:
                context = "Invalid response format."
                helpful_answer = "No helpful answer available."
            
            await websocket.send_text(
                f"Question was: {qs_data} \n\n"
                f"Answer: {helpful_answer} \n\n"
                f"From Context: {context} \n\n"
                )
    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        await websocket.send_text(f"Error during inference: {e}")
        await websocket.close()
        
