from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()

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