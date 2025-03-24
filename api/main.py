from fastapi import FastAPI, HTTPException
import requests
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware

# Cargar modelos globalmente
print("Cargando modelos...")
sentiment_pipeline = pipeline("sentiment-analysis")
rating_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
rating_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
print("Modelos cargados.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://llm_project_frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic models for input and output
class Comment(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    label: str
    score: float

class RegressionResponse(BaseModel):
    rating: float


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
    
@app.post("/classify", response_model=ClassificationResponse)
async def classify_comment(comment: Comment):
    result = sentiment_pipeline(comment.text)[0]
    return ClassificationResponse(label=result["label"], score=result["score"])

# Endpoint de regresión
@app.post("/rate", response_model=RegressionResponse)
async def rate_comment(comment: Comment):
    inputs = rating_tokenizer(comment.text, return_tensors="pt")
    outputs = rating_model(**inputs)
    rating = outputs.logits.item()
    return RegressionResponse(rating=rating)

if __name__ == "__main__":
    import uvicorn
