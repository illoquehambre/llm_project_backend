import asyncio
from contextlib import asynccontextmanager
import requests
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from fastapi.middleware.cors import CORSMiddleware
from datasets import load_dataset
import os
import torch
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cargar modelos al iniciar
    await asyncio.to_thread(load_models)
    yield
    
app = FastAPI(lifespan=lifespan)

# Modelos y tokenizers separados
model_cls = None
model_reg = None
tokenizer_cls = None
tokenizer_reg = None

# Configuración MLFlow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def train_and_save_model(task: str):
    global model_cls, model_reg, tokenizer_cls, tokenizer_reg
    
    logger.info(f"Entrenando modelo para {task}...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Seleccionar y preprocesar el dataset según la tarea
    if task == "classification":
        # Dataset de clasificación: columnas "review" y "label" (0/1)
        raw_dataset = load_dataset("jahjinx/IMDb_movie_reviews")
        # Realizamos el split de train/test
        split_dataset = raw_dataset["train"].train_test_split(test_size=0.2)
        # Limitamos el entrenamiento a 500 filas y test a 200 filas
        train_dataset = split_dataset["train"].select(range(250))
        test_dataset = split_dataset["test"].select(range(100))
        
        # La función de tokenización usa la columna "review"
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
        
        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_test = test_dataset.map(tokenize_function, batched=True)
        
        # Crear el modelo con num_labels=2 para clasificación
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        training_args = TrainingArguments(
            output_dir=f"./results_{task}",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            num_train_epochs=1,  # Reducido para pruebas
            weight_decay=0.01,
        )
        
    elif task == "regression":
        # Dataset de regresión: columnas "review", "rating" y "label"
        raw_dataset = load_dataset("DDDDZQ/imdb_reviews")
        # Realizamos el split de train/test
        split_dataset = raw_dataset["train"].train_test_split(test_size=0.2)
        # Limitamos el entrenamiento a 500 filas y test a 200 filas
        train_dataset = split_dataset["train"].select(range(250))
        test_dataset = split_dataset["test"].select(range(100))
        
        # Si "label" ya contiene lo mismo que "rating", eliminamos "rating" para evitar duplicados
        if "rating" in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns("rating")
        if "rating" in test_dataset.column_names:
            test_dataset = test_dataset.remove_columns("rating")
        
        # Normalizamos las etiquetas de 1-10 a 0-1
        def normalize_labels(example):
            example["label"] = (example["label"] - 1) / 9  # Escalar entre 0 y 1
            return example

        train_dataset = train_dataset.map(normalize_labels)
        test_dataset = test_dataset.map(normalize_labels)

        def tokenize_function(examples):
            return tokenizer(examples["review"], truncation=True, padding="max_length", max_length=256)
        
        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_test = test_dataset.map(tokenize_function, batched=True)

        # Definir función compute_metrics para regresión
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.reshape(-1)  # Asegurar que sean vectores 1D
            labels = labels.reshape(-1)

            # Desnormalizar los valores de 0-1 a 1-10
            predictions = (predictions * 9) + 1
            labels = (labels * 9) + 1

            mse = ((predictions - labels) ** 2).mean()
            return {"mse": mse}  # Retorna el error cuadrático medio
        
        #  Configuramos el modelo para regresión (num_labels=1)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
        training_args = TrainingArguments(
            output_dir=f"./results_{task}",
            eval_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=8,
            num_train_epochs=1,  # Reducido para pruebas
            weight_decay=0.01,
        )
    
    # Preparar el Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics if task == "regression" else None  # Agregar métricas solo en regresión
    )
    
    trainer.train()
    
    # Registrar en MLFlow
    with mlflow.start_run(run_name=f"{task}_model"):
        mlflow.log_params({
            "model_type": "bert-base-uncased",
            "task": task,
            "num_epochs": training_args.num_train_epochs
        })
        
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=f"model_{task}",
            registered_model_name=f"movie_reviews_{task}"
        )
        
    return model, tokenizer

def load_models():
    global model_cls, model_reg, tokenizer_cls, tokenizer_reg
    
    try:
        # Cargar modelo de clasificación desde MLflow
        model_cls = mlflow.pytorch.load_model("models:/movie_reviews_classification/latest")
        tokenizer_cls = AutoTokenizer.from_pretrained("bert-base-uncased")
        logger.info("Modelo de clasificación cargado correctamente")
    except Exception as e:
        logger.error(f"Error cargando modelo clasificación: {str(e)}. Entrenando nuevo...")
        model_cls, tokenizer_cls = train_and_save_model("classification")
    
    try:
        # Cargar modelo de regresión desde MLflow
        model_reg = mlflow.pytorch.load_model("models:/movie_reviews_regression/latest")
        tokenizer_reg = AutoTokenizer.from_pretrained("bert-base-uncased")
        logger.info("Modelo de regresión cargado correctamente")
    except Exception as e:
        logger.error(f"Error cargando modelo regresión: {str(e)}. Entrenando nuevo...")
        model_reg, tokenizer_reg = train_and_save_model("regression")


# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic
class Comment(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    label: str
    score: float

class RegressionResponse(BaseModel):
    rating: float

# Endpoints
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/query/{topic}")
def query_ollama(topic: str):
    try:
        prompt = f"Provide interesting information about {topic}."
        response = requests.post("http://ollama:11434/api/generate", 
        json=
        {
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
    try:
        logger.info(f"model_cls: %s",model_cls)
        logger.info(f"tokenizer_cls: %s",tokenizer_cls)
        inputs = tokenizer_cls(comment.text, return_tensors="pt", truncation=True, max_length=256)
        logger.info(f"inputs: %s",inputs)        
        outputs = model_cls(**inputs)
        logger.info('outputs logs: ', outputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return {
            "label": "Positive" if torch.argmax(probs).item() == 1 else "Negative",
            "score": round(torch.max(probs).item(), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rate", response_model=RegressionResponse)
async def rate_comment(comment: Comment):
    try:
        logger.info(f"model_reg: %s",model_reg)
        logger.info(f"tokenizer_reg: %s",tokenizer_reg)
        inputs = tokenizer_reg(comment.text, return_tensors="pt", truncation=True, max_length=256)
        logger.info(f"inputs: %s",inputs)   
        with torch.no_grad():
            outputs = model_reg(**inputs)
        logger.info('outputs logs: ', outputs)
        # Tomar el logit de salida (ya que el modelo tiene num_labels=1)
        logit = outputs.logits.squeeze().item()  # Extraer el único valor de la salida
        
        # Desnormalizar: convertir de [0,1] a [1,10]
        rating = (logit * 9) + 1  
        
        # Asegurar que el rating no salga fuera del rango
        rating = max(1.0, min(10.0, rating))  # Limitar entre 1 y 10
        
        return {"rating": round(rating, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
