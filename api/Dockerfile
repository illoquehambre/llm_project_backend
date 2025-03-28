FROM python:3.10-slim

WORKDIR /home/app

RUN apt-get update && \
    apt-get install -y g++ git curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt && rm -rf /root/.cache/pip

RUN pip install uvicorn

COPY . .

EXPOSE 8000

ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MLFLOW_TRACKING_USERNAME=mlflow
ENV MLFLOW_TRACKING_PASSWORD=mlflow

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]