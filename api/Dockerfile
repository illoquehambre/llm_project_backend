FROM python:3.10-slim

WORKDIR /home/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt && rm -rf /root/.cache/pip

RUN pip install uvicorn

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]