# -*- coding: utf-8 -*-
"""
Created on Mon Jan 9 11:20:46 2025

@author: mashn
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 15:11:19 2024

@author: mashn
"""
import os
import time
from threading import Thread
from typing import Union
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from model import load_quantized_llm_model, rag_pipeline, connect_weaviate_vector_store, load_retrievers_weaviate, upload_data_weaviate_vector_store



app = FastAPI()

# Load environment variables
WEAVIATE_CLUSTER_URL = str(os.getenv("WEAVIATE_CLUSTER_URL")).strip()
WEAVIATE_API_KEY = str(os.getenv("WEAVIATE_API_KEY")).strip()
HF_TOKEN = str(os.getenv("HF_TOKEN")).strip()
COHERE_API_KEY = str(os.getenv("COHERE_API_KEY")).strip()

if not WEAVIATE_CLUSTER_URL:
    raise ValueError("WEAVIATE_CLUSTER_URL must be set. Follow the cmd syntax set $env:ENV_VARIABLE='AKIA6GBMEVKGEC2WOGW5'")
if not WEAVIATE_API_KEY:
    raise ValueError("WEAVIATE_API_KEY must be set. Follow the cmd syntax set $env:ENV_VARIABLE='AKIA6GBMEVKGEC2WOGW5'")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY must be set. Follow the cmd syntax set $env:ENV_VARIABLE='AKIA6GBMEVKGEC2WOGW5'")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN must be set. Follow the cmd syntax set $env:ENV_VARIABLE='AKIA6GBMEVKGEC2WOGW5'")

print(f"WEAVIATE_CLUSTER_URL:{WEAVIATE_CLUSTER_URL}\n WEAVIATE_API_KEY: {WEAVIATE_API_KEY} \n COHERE_API_KEY: {COHERE_API_KEY} \n HF_TOKEN: {HF_TOKEN}")



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
retriever_and_compression_retriever = {"retriever": None, "compression_retriever": None, "data_base_exists": False}

rag_inference_pipeline = None
weaviate_client = None


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

def start_vector_store_loading():
    global retriever_and_compression_retriever
    global weaviate_client

    
    try:
        weaviate_client, embedding_model = connect_weaviate_vector_store(WEAVIATE_CLUSTER_URL , WEAVIATE_API_KEY, HF_TOKEN)
        print("Weaviate connection successful.")
        if(weaviate_client.collections.exists("HYBRIDRAG")):
            try:
                retriever, compression_retriever = load_retrievers_weaviate(weaviate_client, embedding_model, COHERE_API_KEY=COHERE_API_KEY)
                retriever_and_compression_retriever["retriever"] = retriever
                retriever_and_compression_retriever["compression_retriever"] = compression_retriever
                retriever_and_compression_retriever["data_base_exists"] = True
                print("Retriever loading successful.")
            except Exception as e:
                print(f"Error making document Retrievers: {e}")

        else:
            try:
                upload_data_weaviate_vector_store(weaviate_client, embedding_model)
                retriever, compression_retriever = load_retrievers_weaviate(weaviate_client, embedding_model, COHERE_API_KEY=COHERE_API_KEY)
                retriever_and_compression_retriever["retriever"] = retriever
                retriever_and_compression_retriever["compression_retriever"] = compression_retriever
                retriever_and_compression_retriever["data_base_exists"] = True
                print("Retriever creation and loading successful.")
            except Exception as e:
                print(f"Error Uploading data & making document Retrievers: {e}")
    except Exception as e:
        print(f"Error connecting to weaviate: {e}")


def is_database_ready():
    return retriever_and_compression_retriever["data_base_exists"]


@app.on_event("startup")
async def startup_event():
    global rag_inference_pipeline
    thread_retriever = Thread(target=start_vector_store_loading, daemon=True)
    thread_retriever.start()
    print("Waiting for Retrievers to load...")
    
    time.sleep(3)

    thread_model = Thread(target=start_llm_loading, daemon=True)
    thread_model.start()
    print("Waiting for model to load...")

    while not is_model_ready() or not is_database_ready():
        #print(f"Model ready: {is_model_ready()}, Database ready: {is_database_ready()}")
        time.sleep(1/1000)
        pass
    
    time.sleep(5)
    try:
        # Initialize the RAG pipeline
        rag_inference_pipeline = rag_pipeline(model=model_and_tokenizer["model"], tokenizer=model_and_tokenizer["tokenizer"], retriever=retriever_and_compression_retriever["compression_retriever"])
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
    global weaviate_client
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
        weaviate_client.close()
        await websocket.send_text(f"Error during inference: {e}")
        await websocket.close()
        

