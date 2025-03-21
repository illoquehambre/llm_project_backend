# Usar una imagen base de Python
FROM python:latest

# Establecer el directorio de trabajo
WORKDIR /home/app

# Copiar los archivos de requisitos y la aplicación
COPY requirements.txt ./

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Instalar uvicorn explícitamente
RUN pip install uvicorn

# Copiar el resto de la aplicación
COPY . .

# Exponer el puerto en el que correrá la aplicación
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]