from fastapi import FastAPI, HTTPException
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://llm_project_frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/query/{topic}")
def query_ollama(topic: str):
    try:
        prompt = f"Provide interesting information about {topic}."
        response = requests.post("http://ollama:11434/api/generate", 
        json={
            "model": "qwen2.5:0.5b",
            "stream": False,
            "prompt": prompt
        })
        response.raise_for_status()
        print(response)
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))